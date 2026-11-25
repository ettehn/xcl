//! Partition the optimized compute entry at CFG edges. A dispatch cannot
//! jump back to an earlier dispatch, so a strongly connected component is
//! indivisible. Each crossing edge has its own compact state layout.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::{BinOp, Block, CastOp, Module, Op, TypeDef, Value};
use crate::gpu::dxil::shader::{Feature, RawUav, Shader};

#[derive(Debug)]
pub(crate) struct Edge {
    pub from: Block,
    pub to: Block,
    pub live: Vec<Value>,
}

pub(crate) struct Plan {
    pub blocks: Vec<Vec<Block>>,
    pub edges: Vec<Edge>,
}

#[derive(Clone)]
pub struct DispatchChain {
    pub pieces: Vec<Vec<u8>>,
    pub dwords_per_group: u32,
}

struct Scratch {
    uav: RawUav,
    base: Value,
    bytes_per_slot: u32,
}

impl Scratch {
    fn new(s: &mut Shader, stride: u32, threads: u32, lane: Value) -> Self {
        let uav = s.raw_uav(0, 2);
        let cb = s.cbv(0, 2, 16);
        let zero = s.m.i32c(0);
        let ty = s.m.i32();
        let [width, height, _, _] = s.cbuffer_load_legacy(cb, zero, ty);
        let [x, y, z] = [0, 1, 2].map(|axis| s.group_id(axis));
        let row = s.m.bin(BinOp::Mul, z, height);
        let row = s.m.bin(BinOp::Add, row, y);
        let first = s.m.bin(BinOp::Mul, row, width);
        let group = s.m.bin(BinOp::Add, first, x);
        // Give consecutive logical lanes consecutive words for a slot, including
        // the continuation tag in slot zero. A workgroup owns all its slots.
        let group_words = s.m.i32c(stride * threads);
        let first = s.m.bin(BinOp::Mul, group, group_words);
        let index = s.m.bin(BinOp::Add, first, lane);
        let four = s.m.i32c(4);
        Self { uav, base: s.m.bin(BinOp::Mul, index, four), bytes_per_slot: threads * 4 }
    }

    /// Every logical thread participates, including threads whose guest path
    /// skipped this piece. Exactly one thread owns each shared word.
    fn transfer_shared(&self, s: &mut Shader, lane: Value, lanes: u32, arrays: &[(Value, u32, u32)], restore: bool) {
        use crate::gpu::dxil::ops::barrier::GROUP_SYNC;
        if !restore { s.barrier(GROUP_SYNC); }
        for &(global, count, slot) in arrays {
            for page in 0..count.div_ceil(lanes) {
                let remaining = count - page * lanes;
                let mut join = None;
                if remaining < lanes {
                    let n = s.m.i32c(remaining);
                    let active = s.m.cmp(super::Pred::ULt, lane, n);
                    let body = s.m.new_block();
                    let next = s.m.new_block();
                    s.m.cond_br(active, body, next);
                    s.m.set_block(body);
                    join = Some(next);
                }
                let zero = s.m.i32c(0);
                let base = s.m.i32c(page * lanes);
                let index = s.m.bin(BinOp::Add, base, lane);
                let ptr = s.m.gep(global, &[zero, index]);
                let addr = self.at(&mut s.m, slot + page);
                if restore {
                    let v = s.raw_load(self.uav, addr, 1)[0];
                    s.m.store(ptr, v, 4);
                } else {
                    let v = s.m.load(ptr, 4);
                    s.raw_store(self.uav, addr, &[v]);
                }
                if let Some(next) = join {
                    s.m.br(next);
                    s.m.set_block(next);
                }
            }
        }
        if restore { s.barrier(GROUP_SYNC); }
    }

    fn at(&self, m: &mut Module, slot: u32) -> Value {
        let offset = m.i32c(slot * self.bytes_per_slot);
        m.bin(BinOp::Add, self.base, offset)
    }
}

impl Shader {
    /// Native wave32 uses the hardware lane in a single full wave. Split
    /// wave64 recreates the flat console thread ID on every dispatch, including
    /// when the hardware schedules the waves in a different order.
    pub(crate) fn split_compute(&self, threads: [u32; 3], split_tid: Option<Value>, limit: usize) -> Option<DispatchChain> {
        let lanes: u32 = threads.into_iter().try_fold(1u32, u32::checked_mul)?;
        let multi = split_tid.is_some() && lanes > 64;
        if self.m.globals.iter().any(|g| !g.constant
            && !(split_tid.is_some() && matches!(g.name.as_str(), "xcl_xw" | "xcl_xs"))
            && !(multi && g.name == "xcl_xc"))
        {
            return None;
        }
        if self.m.blocks.iter().map(Vec::len).sum::<usize>() <= limit { return None; }
        let mut prefix = Vec::new();
        if let Some(tid) = split_tid {
            let end = self.m.blocks[0].iter().position(|&i| Value::Inst(i) == tid)?;
            prefix.extend_from_slice(&self.m.blocks[0][..=end]);
        }
        let mut consumed = HashSet::new();
        for &i in self.m.blocks.iter().flatten() {
            self.m.insts[i as usize].op.for_each_operand(|v| { consumed.insert(v); });
        }
        let root_is = |v, name| self.m.global_root(v).is_some_and(|g| self.m.globals[g].name == name);
        let spins: HashSet<_> = self.m.spin_regions.iter().flatten().copied().collect();
        for (b, block) in self.m.blocks.iter().enumerate() {
            // In a single split console wave each inter-half exchange begins
            // with TGSM stores, one rendezvous, peer loads and a rendezvous.
            // A block ending anywhere inside that sequence cannot be a cut.
            let mut exchange = 0;
            for &i in block {
                if b == 0 && prefix.contains(&i) { continue; }
                match &self.m.insts[i as usize].op {
                    Op::AtomicRmw(super::RmwOp::Add, p, _) if multi && root_is(*p, "xcl_xc") => {}
                    Op::Load(p, _) | Op::Store(p, _, _) if multi && root_is(*p, "xcl_xs") => {}
                    Op::Call(f, args) if multi && spins.contains(&Block(b as u32))
                        && self.m.funcs[*f as usize].name == "dx.op.barrier"
                        && self.m.const_u64(args[1]) == Some(crate::gpu::dxil::ops::barrier::TGSM_FENCE as u64) => {}
                    Op::Alloca(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => return None,
                    Op::Store(..) if split_tid.is_some() && !multi && exchange <= 1 => exchange = 1,
                    Op::Load(p, _) if self.m.global_root(*p).is_some_and(|g| self.m.globals[g].constant) => {}
                    Op::Load(..) if split_tid.is_some() && !multi && matches!(exchange, 2 | 3) => exchange = 3,
                    Op::Load(..) | Op::Store(..) => return None,
                    Op::Call(f, args) if self.m.funcs[*f as usize].name == "dx.op.barrier"
                            && self.m.const_u64(args[1]) == Some(crate::gpu::dxil::ops::barrier::GROUP_SYNC as u64) => {
                        exchange = match exchange { 1 => 2, 3 => 0, _ => return None };
                    }
                    // Sub-dword stores use interlocked AND/OR. They return an
                    // old word that nobody observes; every write stays in place.
                    Op::Call(f, args) if self.m.funcs[*f as usize].name == "dx.op.atomicBinOp.i32"
                        && matches!(self.m.const_u64(args[2]), Some(1 | 2))
                        && !consumed.contains(&Value::Inst(i)) => {}
                    Op::Call(f, _) if ["dx.op.barrier", "dx.op.atomic"].iter().any(|p| self.m.funcs[*f as usize].name.starts_with(p)) => return None,
                    _ => {}
                }
            }
            if exchange != 0 { return None; }
        }
        let plan = self.m.dispatch_plan(limit)?;
        let mut stride = 1;
        for edge in &plan.edges {
            let mut words = 1;
            for &v in &edge.live {
                words += self.m.transport_words(v)?;
            }
            stride = stride.max(words);
        }
        let mut shared = Vec::new();
        if multi {
            for (i, g) in self.m.globals.iter().enumerate().filter(|(_, g)| matches!(g.name.as_str(), "xcl_xc" | "xcl_xs")) {
                let TypeDef::Array(n, _) = self.m.types[g.ty.0 as usize] else { return None };
                shared.push((Value::Global(i as u32), u32::try_from(n).ok()?, stride));
                stride = stride.checked_add(u32::try_from(n).ok()?.div_ceil(lanes))?;
            }
        }
        let pieces = (0..plan.blocks.len()).map(|i| self.make_piece(&plan, i, stride, threads, &prefix, &shared)).collect::<Option<Vec<_>>>()?;
        Some(DispatchChain { pieces, dwords_per_group: stride.checked_mul(lanes)? })
    }

    fn make_piece(&self, plan: &Plan, stage: usize, stride: u32, threads: [u32; 3], prefix: &[u32], shared: &[(Value, u32, u32)]) -> Option<Vec<u8>> {
        let src = &self.m;
        let mut dst = self.dispatch_piece();
        let mut values = HashMap::new();
        for &id in prefix {
            let def = &src.insts[id as usize];
            let mut op = def.op.clone();
            op.map_operands(|v| *values.get(&v).unwrap_or(&v));
            values.insert(Value::Inst(id), dst.m.push(op, def.ty));
        }
        let lane = prefix.last().map(|&i| values[&Value::Inst(i)]).unwrap_or_else(|| dst.wave_lane_index());
        let lanes = threads.iter().product();
        let scratch = Scratch::new(&mut dst, stride, lanes, lane);
        if stage > 0 && !shared.is_empty() {
            scratch.transfer_shared(&mut dst, lane, lanes, shared, true);
        }
        let dispatch_entry = dst.m.current_block();
        let local: HashMap<_, _> = plan.blocks[stage].iter().map(|&b| (b, dst.m.new_block())).collect();
        let mut hops = HashMap::new();
        let mut exits = HashMap::new();
        let mut cases = Vec::new();
        let mut paths = Vec::new();
        let mut incoming = std::collections::BTreeMap::<Value, Vec<(Vec<Value>, Block)>>::new();
        let mut statics = BTreeSet::new();
        let merge = dst.m.new_block();
        let done = dst.m.new_block();
        let selector = if stage != 0 { Some(dst.raw_load(scratch.uav, scratch.base, 1)[0]) } else { None };
        for (index, edge) in plan.edges.iter().enumerate() {
            let tag = dst.m.i32c(index as u32 + 1);
            if local.contains_key(&edge.from) {
                exits.insert((edge.from, edge.to), (dst.m.new_block(), tag));
            }
            if let Some(&target) = local.get(&edge.to) {
                let bridge = dst.m.new_block();
                dst.m.set_block(bridge);
                let mut slot = 1;
                for &v in &edge.live {
                    let n = src.transport_words(v)?;
                    if n == 0 {
                        statics.insert(v);
                        continue;
                    }
                    let words = (slot..slot + n).map(|word| {
                        let at = scratch.at(&mut dst.m, word);
                        dst.raw_load(scratch.uav, at, 1)[0]
                    }).collect();
                    incoming.entry(v).or_default().push((words, bridge));
                    slot += n;
                }
                dst.m.br(merge);
                let hop = dst.m.new_block();
                dst.m.set_block(hop);
                dst.m.br(target);
                cases.push((tag, bridge));
                paths.push((bridge, tag, hop));
                hops.insert((edge.from, edge.to), hop);
            }
        }
        dst.m.set_block(dispatch_entry);
        if let Some(sel) = selector {
            dst.m.switch(sel, done, &cases);
            dst.m.set_block(merge);
            let word = dst.m.i32();
            let mut unpack = Vec::new();
            for (original, inc) in incoming {
                let mut words = Vec::new();
                for k in 0..inc[0].0.len() {
                    let phi = dst.m.phi(word);
                    for &(pred, _, _) in &paths {
                        let v = inc.iter().find(|p| p.1 == pred).map(|p| p.0[k]).unwrap_or_else(|| dst.m.undef(word));
                        dst.m.add_incoming(phi, v, pred);
                    }
                    words.push(phi);
                }
                unpack.push((original, words));
            }
            for (original, words) in unpack {
                values.insert(original, src.unpack(original, &words, &mut dst.m));
            }
            for handle in statics {
                src.rebuild_static(handle, &mut dst.m, &mut values)?;
            }
            let routes: Vec<_> = paths.iter().map(|&(_, tag, hop)| (tag, hop)).collect();
            dst.m.switch(sel, done, &routes);
        } else {
            dst.m.br(local[&Block(0)]);
        }
        dst.m.set_block(done);
        let checkpoint = !shared.is_empty() && stage + 1 < plan.blocks.len();
        if checkpoint { scratch.transfer_shared(&mut dst, lane, lanes, shared, false); }
        dst.m.ret();

        // Assign all instruction identities before fixing operands. A loop's
        // phi can use an instruction in a block copied later.
        let mut copied = Vec::new();
        for &block in &plan.blocks[stage] {
            dst.m.set_block(local[&block]);
            for &i in &src.blocks[block.0 as usize] {
                if block == Block(0) && prefix.contains(&i) { continue; }
                let old = &src.insts[i as usize];
                if matches!(old.op, Op::Ret) && stage + 1 < plan.blocks.len() {
                    let zero = dst.m.i32c(0);
                    dst.raw_store(scratch.uav, scratch.base, &[zero]);
                    if checkpoint { dst.m.br(done); continue; }
                }
                let v = dst.m.push(old.op.clone(), old.ty);
                values.insert(Value::Inst(i), v);
                copied.push((block, inst(v)?));
            }
        }
        for (block, id) in copied {
            let op = &mut dst.m.insts[id].op;
            let mut complete = true;
            op.map_operands(|v| match v {
                Value::Inst(_) => values.get(&v).copied().unwrap_or_else(|| { complete = false; v }),
                _ => v,
            });
            if !complete { return None; }
            let target = |b: Block| local.get(&b).copied().or_else(|| exits.get(&(block, b)).map(|p| p.0));
            match op {
                Op::Phi(inc) => for (_, p) in inc {
                    *p = *local.get(p).or_else(|| hops.get(&(*p, block)))?;
                },
                Op::Br(b) => *b = target(*b)?,
                Op::CondBr(_, t, f) => { *t = target(*t)?; *f = target(*f)?; }
                Op::Switch(_, d, cases) => {
                    *d = target(*d)?;
                    for (_, t) in cases { *t = target(*t)?; }
                }
                _ => {}
            }
        }
        for edge in &plan.edges {
            let Some(&(b, tag)) = exits.get(&(edge.from, edge.to)) else { continue };
            dst.m.set_block(b);
            let mut slot = 1;
            for &v in &edge.live {
                let n = src.transport_words(v)?;
                if n != 0 {
                    let words = src.pack(v, *values.get(&v)?, &mut dst.m);
                    for (k, word) in words.into_iter().enumerate() {
                        let at = scratch.at(&mut dst.m, slot + k as u32);
                        dst.raw_store(scratch.uav, at, &[word]);
                    }
                    slot += n;
                }
            }
            dst.raw_store(scratch.uav, scratch.base, &[tag]);
            if checkpoint { dst.m.br(done); } else { dst.m.ret(); }
        }
        dst.infer_piece_features();
        // a refused piece leaves the whole program unsplit: say so
        let refused = |e: String| crate::xlog_warn!("compute split: piece {} of {} refused, the program stays whole: {}",
            stage + 1, plan.blocks.len(), e.lines().next().unwrap_or_default());
        let blob = dst.try_finish(threads, Some(32)).map_err(|e| refused(e.to_string())).ok()?;
        crate::gpu::dxil::validate::validate(&blob).map_err(refused).ok()?;
        Some(blob)
    }

    fn infer_piece_features(&mut self) {
        for f in [Feature::Doubles, Feature::Int64Ops, Feature::DoubleExtensions,
                  Feature::ResourceHeapIndexing, Feature::SamplerHeapIndexing] {
            self.features.remove(&f);
        }
        // typed UAV loads (`texture_load` without a mip): the whole program's flag stays only on
        // the pieces that have one, the validator wants flags to match usage
        let uav_load = |args: &[Value], m: &Module| args.get(2).is_some_and(|&mip| matches!(mip, Value::Const(c) if matches!(m.consts[c as usize], super::ConstDef::Undef(_))));
        if !self.m.blocks.iter().flatten().any(|&i| matches!(&self.m.insts[i as usize].op,
            Op::Call(f, args) if self.m.funcs[*f as usize].name.starts_with("dx.op.textureLoad") && uav_load(args, &self.m))) {
            self.features.remove(&Feature::TypedUavLoadAdditionalFormats);
        }
        for def in &self.m.insts {
            if let Op::Call(f, args) = &def.op {
                if self.m.funcs[*f as usize].name == "dx.op.createHandleFromHeap" {
                    self.features.insert(if self.m.const_u64(args[2]) == Some(0) {
                        Feature::ResourceHeapIndexing
                    } else { Feature::SamplerHeapIndexing });
                }
            }
        }
    }
}

fn inst(v: Value) -> Option<usize> {
    if let Value::Inst(i) = v { Some(i as usize) } else { None }
}

impl Module {
    fn global_root(&self, v: Value) -> Option<usize> {
        match v {
            Value::Global(g) => Some(g as usize),
            Value::Inst(i) => match &self.insts[i as usize].op {
                Op::Gep(p, _) => self.global_root(*p),
                _ => None,
            },
            _ => None,
        }
    }

    /// Only an index is transported. Resource handles and TGSM pointers are
    /// created inside the receiving dispatch, using their original properties.
    fn indexed(&self, v: Value) -> Option<Value> {
        let op = &self.insts[inst(v)?].op;
        if let Op::Gep(Value::Global(g), indices) = op {
            if matches!(self.globals[*g as usize].name.as_str(), "xcl_xs" | "xcl_xc") && indices.len() == 2 && self.const_u64(indices[0]) == Some(0) {
                return Some(indices[1]);
            }
        }
        if let Op::Call(f, a) = op {
            if self.funcs[*f as usize].name == "dx.op.annotateHandle" && matches!(a[2], Value::Const(_)) {
                if let Op::Call(create, args) = &self.insts[inst(a[1])?].op {
                    if self.funcs[*create as usize].name == "dx.op.createHandleFromHeap" && args[2..].iter().all(|v| matches!(v, Value::Const(_))) {
                        return Some(args[1]);
                    }
                }
            }
        }
        None
    }

    fn transport_words(&self, v: Value) -> Option<u32> {
        let def = &self.insts[inst(v)?];
        match &self.types[def.ty.0 as usize] {
            TypeDef::Int(1..=32) | TypeDef::Float => Some(1),
            TypeDef::Int(64) => Some(2),
            TypeDef::Struct(..) if self.static_handle(v) => Some(0),
            _ if self.indexed(v).is_some() => Some(1),
            _ => None,
        }
    }

    fn static_handle(&self, v: Value) -> bool {
        let Value::Inst(i) = v else { return true };
        let Op::Call(f, args) = &self.insts[i as usize].op else { return false };
        match self.funcs[*f as usize].name.as_str() {
            "dx.op.createHandleFromBinding" => args.iter().all(|v| matches!(v, Value::Const(_))),
            "dx.op.annotateHandle" => args.iter().all(|&v| self.static_handle(v)),
            _ => false,
        }
    }

    fn rebuild_static(&self, v: Value, dst: &mut Module, map: &mut HashMap<Value, Value>) -> Option<Value> {
        let Value::Inst(id) = v else { return Some(v) };
        if let Some(&old) = map.get(&v) { return Some(old); }
        let Op::Call(f, args) = &self.insts[id as usize].op else { return None };
        let args: Vec<_> = args.iter().map(|&a| self.rebuild_static(a, dst, map)).collect::<Option<_>>()?;
        let new = dst.call(*f, &args);
        map.insert(v, new);
        Some(new)
    }

    fn pack(&self, original: Value, v: Value, dst: &mut Module) -> Vec<Value> {
        if let Some(index) = dst.indexed(v) { return vec![index]; }
        let ty = self.insts[inst(original).unwrap()].ty;
        let word = dst.i32();
        match self.types[ty.0 as usize] {
            TypeDef::Int(32) => vec![v],
            TypeDef::Int(64) => {
                let shift = dst.const_int(ty, 32);
                let high = dst.bin(BinOp::LShr, v, shift);
                vec![dst.cast(CastOp::Trunc, v, word), dst.cast(CastOp::Trunc, high, word)]
            }
            TypeDef::Float => vec![dst.bitcast(v, word)],
            _ => vec![dst.cast(CastOp::ZExt, v, word)],
        }
    }

    fn unpack(&self, original: Value, words: &[Value], dst: &mut Module) -> Value {
        if self.indexed(original).is_some() {
            match &self.insts[inst(original).unwrap()].op {
                Op::Gep(base, indices) => return dst.gep(*base, &[indices[0], words[0]]),
                Op::Call(f, annotate) => {
                    let Op::Call(create, args) = &self.insts[inst(annotate[1]).unwrap()].op else { unreachable!() };
                    let handle = dst.call(*create, &[args[0], words[0], args[2], args[3]]);
                    return dst.call(*f, &[annotate[0], handle, annotate[2]]);
                }
                _ => unreachable!(),
            }
        }
        let ty = self.insts[inst(original).unwrap()].ty;
        match self.types[ty.0 as usize] {
            TypeDef::Int(32) => words[0],
            TypeDef::Int(64) => {
                let low = dst.cast(CastOp::ZExt, words[0], ty);
                let high = dst.cast(CastOp::ZExt, words[1], ty);
                let shift = dst.const_int(ty, 32);
                let shifted = dst.bin(BinOp::Shl, high, shift);
                dst.bin(BinOp::Or, low, shifted)
            }
            TypeDef::Float => dst.bitcast(words[0], ty),
            _ => dst.cast(CastOp::Trunc, words[0], ty),
        }
    }
}

impl Module {
    /// Preserve symbol and type indices when copying instructions into a
    /// piece. The destination has its own blocks, instructions and metadata.
    pub(in crate::gpu::dxil) fn declarations(&self) -> Self {
        Self {
            types: self.types.clone(),
            type_ids: self.type_ids.clone(),
            consts: self.consts.clone(),
            const_ids: self.const_ids.clone(),
            funcs: self.funcs.clone(),
            func_ids: self.func_ids.clone(),
            globals: self.globals.clone(),
            ..Self::new()
        }
    }

    pub(crate) fn dispatch_plan(&self, limit: usize) -> Option<Plan> {
        let (order, _) = self.block_layout();
        let count = |b: usize| self.blocks[b].len();
        if limit == 0 || order.iter().copied().map(count).sum::<usize>() <= limit {
            return None;
        }
        let mut predecessors = vec![Vec::new(); self.blocks.len()];
        for &b in &order {
            for target in self.successors(b) {
                predecessors[target.0 as usize].push(b);
            }
        }

        // In reverse postorder of the original graph, flooding the transpose
        // visits one component at a time in topological order (Kosaraju).
        let mut visited = vec![false; self.blocks.len()];
        let mut blocks = vec![Vec::new()];
        let mut stages = vec![usize::MAX; self.blocks.len()];
        let mut components = Vec::new();
        for &root in &order {
            if visited[root] {
                continue;
            }
            let mut component = Vec::new();
            let mut cost = 0;
            let mut stack = vec![root];
            visited[root] = true;
            while let Some(b) = stack.pop() {
                cost += count(b);
                component.push(b);
                for &p in &predecessors[b] {
                    if !visited[p] {
                        visited[p] = true;
                        stack.push(p);
                    }
                }
            }
            for &b in &component {
                stages[b] = components.len();
            }
            components.push((cost, component));
        }
        // A wait polling the peer's counter cannot complete if the peer's
        // signal was deferred to another dispatch. Pack these SCCs together.
        let mut through: Vec<usize> = (0..components.len()).collect();
        for region in &self.spin_regions {
            let mut span = region.iter().map(|b| stages[b.0 as usize]).filter(|&i| i != usize::MAX);
            if let Some(i) = span.next() {
                let (lo, hi) = span.fold((i, i), |(lo, hi), i| (lo.min(i), hi.max(i)));
                through[lo] = through[lo].max(hi);
            }
        }
        let mut i = 0;
        let mut used = 0;
        while i < components.len() {
            let first = i;
            let mut last = i;
            let mut cost = 0;
            while i <= last {
                last = last.max(through[i]);
                cost += components[i].0;
                i += 1;
            }
            if cost > limit { return None; }
            if used + cost > limit {
                blocks.push(Vec::new());
                used = 0;
            }
            used += cost;
            for (_, component) in &components[first..i] {
                for &b in component { stages[b] = blocks.len() - 1; }
            }
        }

        let mut crossing = BTreeSet::new();
        for &b in &order {
            let from = Block(b as u32);
            blocks[stages[b]].push(from);
            for to in self.successors(b) {
                if stages[b] != stages[to.0 as usize] {
                    debug_assert!(stages[b] < stages[to.0 as usize]);
                    crossing.insert((from, to));
                }
            }
        }
        let mut live_in = vec![HashSet::new(); self.blocks.len()];
        let mut phi_input = HashMap::<_, BTreeSet<_>>::new();
        let mut work = Vec::new();
        for &b in &order {
            for &inst in &self.blocks[b] {
                let op = &self.insts[inst as usize].op;
                // A phi reads each operand on its incoming edge, including
                // when the definition is its predecessor's last instruction.
                if let Op::Phi(inputs) = op {
                    for &(value, pred) in inputs {
                        if let Value::Inst(i) = value {
                            let edge = (pred, Block(b as u32));
                            if crossing.contains(&edge) {
                                phi_input.entry(edge).or_default().insert(value);
                            }
                            if self.def_block[i as usize] != pred {
                                work.push((pred.0 as usize, i));
                            }
                        }
                    }
                } else {
                    op.for_each_operand(|v| {
                        if let Value::Inst(i) = v {
                            if self.def_block[i as usize].0 as usize != b {
                                work.push((b, i));
                            }
                        }
                    });
                }
            }
        }
        while let Some((b, value)) = work.pop() {
            if live_in[b].insert(value) {
                for &p in &predecessors[b] {
                    if self.def_block[value as usize].0 as usize != p {
                        work.push((p, value));
                    }
                }
            }
        }
        let edges = crossing.into_iter().map(|(from, to)| {
            let mut live = phi_input.remove(&(from, to)).unwrap_or_default();
            live.extend(live_in[to.0 as usize].iter().copied().map(Value::Inst));
            Edge { from, to, live: live.into_iter().collect() }
        }).collect();
        Some(Plan { blocks, edges })
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::{
        dxil::{kernel::{Entry, KernelOptions}, lower::Lw, module::{BinOp, CastOp, Pred}},
        recomp::{runner, CsInterface},
    };

    fn compare(threads: u32) {
        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        let geometry = [threads, 1, 1];
        let split = threads == 64;
        let cs = CsInterface { threads: geometry, ..Default::default() };
        let options = KernelOptions { wave64: split, split, mem_windows: 1, sampler_base: 0 };
        let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let mem = lw.mems[0];
        let group = lw.s.group_id(0);
        let count = lw.c(threads);
        let base = lw.mul(group, count);
        let index = lw.add(base, tid);
        let offset = lw.shlc(index, 2);
        let original = lw.s.raw_load(mem, offset, 1)[0];
        // Two different dynamic dwords cross every cut as one i64. With
        // slot-major scratch the high word is a wave width past the low word.
        let i64_ty = lw.s.m.i64();
        let lo = lw.s.m.cast(CastOp::ZExt, original, i64_ty);
        let hi = lw.s.m.cast(CastOp::ZExt, tid, i64_ty);
        let shift = lw.s.m.const_int(i64_ty, 32);
        let hi = lw.s.m.bin(BinOp::Shl, hi, shift);
        let pair = lw.s.m.bin(BinOp::Or, lo, hi);
        let trips = lw.andc(tid, 3);
        let trips = lw.addc(trips, 1);
        let mut value = original;
        let i32 = lw.s.m.i32();
        let zero = lw.c(0);

        // Each loop both consumes the previous loop's result and reuses
        // the original per-lane input. All lanes eventually reconverge.
        for round in 1..=16 {
            let pred = lw.cur();
            let body = lw.s.m.new_block();
            let exit = lw.s.m.new_block();
            lw.s.m.br(body);
            lw.s.m.set_block(body);
            let counter = lw.s.m.phi(i32);
            let acc = lw.s.m.phi(i32);
            lw.s.m.add_incoming(counter, zero, pred);
            lw.s.m.add_incoming(acc, value, pred);
            let next = lw.addc(counter, 1);
            value = lw.add(acc, original);
            value = lw.addc(value, round);
            lw.s.m.add_incoming(counter, next, body);
            lw.s.m.add_incoming(acc, value, body);
            let again = lw.cmp(Pred::ULt, next, trips);
            lw.s.m.cond_br(again, body, exit);
            lw.s.m.set_block(exit);
            if split && round == 8 {
                let other_half = lw.exchange(lw.half);
                value = lw.add(value, other_half);
            }
        }
        let pair_lo = lw.s.m.cast(CastOp::Trunc, pair, i32);
        let high = lw.s.m.bin(BinOp::LShr, pair, shift);
        let pair_hi = lw.s.m.cast(CastOp::Trunc, high, i32);
        let both = lw.xor(pair_lo, pair_hi);
        value = lw.add(value, both);
        lw.s.raw_store(mem, offset, &[value]);
        lw.s.m.ret();
        lw.s.m.optimize();
        let chain = lw.s.split_compute(geometry, split.then_some(tid), 40);
        let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
        crate::gpu::dxil::validate::validate(&whole).expect("valid unsplit fixture");
        let chain = chain.expect("split the synthetic program");
        assert!(chain.pieces.len() >= 3);
        let initial: Vec<_> = (1..=threads * 3).collect();
        let expected: Vec<_> = initial.iter().enumerate().map(|(i, &v)| {
            let n = (i as u32 % 4) + 1;
            v + n * (16 * v + 136) + u32::from(split && i as u32 % threads < 32) + (v ^ (i as u32 % threads))
        }).collect();
        let reference = runner::dispatch(&whole, [3, 1, 1], &[], &initial).unwrap();
        assert_eq!(reference, expected);
        let result = runner::dispatch_split(&chain, [3, 1, 1], &initial).unwrap();
        assert_eq!(result, reference);
    }

    /// A typed UAV load in one piece only: the whole program declares its flag, the other pieces
    /// must not (the validator wants flags to match usage, and a refused piece leaves it whole).
    #[test]
    fn dispatch_chain_declares_typed_uav_loads_only_where_they_are() {
        use crate::gpu::dxil::ops::{CompType, TexKind};

        let geometry = [32, 1, 1];
        let options = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
        let cs = CsInterface { threads: geometry, ..Default::default() };
        let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let mem = lw.mems[0];
        let offset = lw.shlc(tid, 2);
        let first = lw.s.raw_load(mem, offset, 1)[0];
        let entry = lw.cur();
        let middle = lw.s.m.new_block();
        let last = lw.s.m.new_block();
        let width = lw.s.wave_lane_count();
        let taken = lw.cmp(Pred::ULt, tid, width);
        lw.s.m.cond_br(taken, middle, last);
        lw.s.m.set_block(middle);
        let slot = lw.s.m.i32c(9);
        let texture = lw.s.heap_texture(slot, TexKind::D2, true, CompType::F32, 4, false);
        lw.s.typed_uav_load_additional_formats();
        let texel = lw.s.texture_load(texture, None, &[tid, tid], &[]);
        let i32 = lw.s.m.i32();
        let bits = lw.s.m.cast(CastOp::BitCast, texel[1], i32);
        let loaded = lw.xor(first, bits);
        lw.s.m.br(last);
        lw.s.m.set_block(last);
        let value = lw.s.m.phi(i32);
        lw.s.m.add_incoming(value, loaded, middle);
        lw.s.m.add_incoming(value, first, entry);
        lw.s.raw_store(mem, offset, &[value]);
        lw.s.m.ret();
        lw.s.m.optimize();
        let limit = lw.s.m.blocks.iter().map(Vec::len).max().unwrap();
        let chain = lw.s.split_compute(geometry, None, limit).expect("every piece validates");
        assert!(chain.pieces.len() >= 2);
    }

    #[test]
    fn dispatch_chain_four_wave64s_preserves_partner_epochs() {
        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        let geometry = [16, 16, 1];
        let cs = CsInterface { threads: geometry, ..Default::default() };
        let options = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
        let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let half = lw.lshrc(tid, 5);
        let group = lw.s.group_id(0);
        let group_base = lw.shlc(group, 8);
        let index = lw.add(group_base, tid);
        let addr = lw.shlc(index, 2);
        let mem = lw.mems[0];
        let initial = lw.s.raw_load(mem, addr, 1)[0];
        let acc = lw.tmp_var();
        lw.wr(acc, initial);
        let odd_wave = lw.andc(tid, 64);
        let odd_wave = lw.nz(odd_wave);
        // Only odd guest waves perform every exchange. A single common
        // exchange count for all guest waves would corrupt their results.
        for round in 1..=12 {
            let mut exchange = |lw: &mut Lw| {
                let payload = lw.addc(half, round);
                let peer = lw.exchange(payload);
                let old = lw.rd(acc);
                let sum = lw.add(old, peer);
                lw.wr(acc, sum);
                Ok(())
            };
            if round % 2 == 1 { lw.guarded(odd_wave, &mut exchange).unwrap(); }
            else { exchange(&mut lw).unwrap(); }
            let next = lw.block(Some(1));
            lw.br(next);
            lw.s.m.set_block(next);
        }
        let v = lw.rd(acc);
        lw.s.raw_store(mem, addr, &[v]);
        lw.s.m.ret();
        lw.s.m.optimize();
        let chain = lw.s.split_compute(geometry, Some(tid), 160).expect("split four guest waves");
        assert!(chain.pieces.len() >= 3);
        let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
        let input: Vec<u32> = (1..=512).collect();
        let expected: Vec<_> = input.iter().enumerate().map(|(i, v)| {
            let tid = i as u32 % 256;
            v + (1..=12).filter(|round| round % 2 == 0 || tid & 64 != 0)
                .map(|round| ((tid / 32) ^ 1) + round).sum::<u32>()
        }).collect();
        let groups = [2, 1, 1];
        let reference = runner::dispatch(&whole, groups, &[], &input).unwrap();
        assert_eq!(reference, expected);
        assert_eq!(runner::dispatch_split(&chain, groups, &input).unwrap(), reference);
    }

    #[test]
    fn dispatch_chain_wave32_live_loops() { compare(32); }

    /// A guest wave64 split across two hardware wave32s.
    #[test]
    fn dispatch_chain_wave64_live_loops_and_exchange() { compare(64); }
    /// Early exits and forward skips retain their own continuation tags; group
    /// indexing must keep every lane separate across all three dispatch axes.
    #[test]
    fn dispatch_chain_divergent_forward_edges_3d() {
        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        println!("CHAIN_GPU={}", runner::adapter().unwrap());
        for threads in [32, 64] {
            let geometry = [threads, 1, 1];
            let split = threads == 64;
            let cs = CsInterface { threads: geometry, ..Default::default() };
            let options = KernelOptions { wave64: split, split, mem_windows: 1, sampler_base: 0 };
            let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
            let (tid, _) = lw.setup_lanes_compute(geometry, false);
            let mem = lw.mems[0];
            let [x, y, z] = [0, 1, 2].map(|axis| lw.s.group_id(axis));
            let three = lw.c(3);
            let row = lw.mul(z, three);
            let row = lw.add(row, y);
            let row = lw.shlc(row, 1);
            let group = lw.add(row, x);
            let count = lw.c(threads);
            let base = lw.mul(group, count);
            let index = lw.add(base, tid);
            let offset = lw.shlc(index, 2);
            let original = lw.s.raw_load(mem, offset, 1)[0];
            let mode = lw.andc(tid, 3);
            let trips = lw.addc(mode, 1);
            let zero = lw.c(0);
            let one = lw.c(1);
            let i32 = lw.s.m.i32();
            let early = lw.s.m.new_block();
            let choose = lw.s.m.new_block();
            let first = lw.s.m.new_block();
            let middle = lw.s.m.new_block();
            let second = lw.s.m.new_block();
            let done = lw.s.m.new_block();
            let exits = lw.cmp(Pred::Eq, mode, zero);
            lw.s.m.cond_br(exits, early, choose);
            lw.s.m.set_block(early);
            let marker = lw.addc(index, 100_000);
            lw.s.raw_store(mem, offset, &[marker]);
            lw.s.m.ret();
            lw.s.m.set_block(choose);
            let skips = lw.cmp(Pred::Eq, mode, one);
            lw.s.m.cond_br(skips, middle, first);
            lw.s.m.set_block(first);
            let mut value = original;
            let mut middle_value = None;
            for round in 1..=16 {
                if round == 9 {
                    let previous = lw.cur();
                    lw.s.m.br(middle);
                    lw.s.m.set_block(middle);
                    let joined = lw.s.m.phi(i32);
                    lw.s.m.add_incoming(joined, original, choose);
                    lw.s.m.add_incoming(joined, value, previous);
                    value = joined;
                    middle_value = Some(joined);
                    let stops = lw.cmp(Pred::Eq, mode, three);
                    lw.s.m.cond_br(stops, done, second);
                    lw.s.m.set_block(second);
                }
                let pred = lw.cur();
                let body = lw.s.m.new_block();
                let exit = lw.s.m.new_block();
                lw.s.m.br(body);
                lw.s.m.set_block(body);
                let counter = lw.s.m.phi(i32);
                let acc = lw.s.m.phi(i32);
                lw.s.m.add_incoming(counter, zero, pred);
                lw.s.m.add_incoming(acc, value, pred);
                let next = lw.addc(counter, 1);
                value = lw.add(acc, original);
                value = lw.addc(value, round);
                lw.s.m.add_incoming(counter, next, body);
                lw.s.m.add_incoming(acc, value, body);
                let again = lw.cmp(Pred::ULt, next, trips);
                lw.s.m.cond_br(again, body, exit);
                lw.s.m.set_block(exit);
            }
            let last = lw.cur();
            lw.s.m.br(done);
            lw.s.m.set_block(done);
            let final_value = lw.s.m.phi(i32);
            lw.s.m.add_incoming(final_value, middle_value.unwrap(), middle);
            lw.s.m.add_incoming(final_value, value, last);
            lw.s.raw_store(mem, offset, &[final_value]);
            lw.s.m.ret();
            lw.s.m.optimize();
            let plan = lw.s.m.dispatch_plan(64).expect("partition branch fixture");
            let stage = |b| plan.blocks.iter().position(|blocks| blocks.contains(&b)).unwrap();
            assert!(plan.edges.iter().any(|edge| stage(edge.to) > stage(edge.from) + 1), "exercise a forward skip over an intermediate piece");
            let chain = lw.s.split_compute(geometry, split.then_some(tid), 64).expect("split branch fixture");
            assert!(chain.pieces.len() >= 3);
            let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
            crate::gpu::dxil::validate::validate(&whole).expect("valid unsplit fixture");
            let words = (threads * 12) as usize;
            let initial: Vec<_> = (0..words + 16).map(|i| if i < words { i as u32 + 17 } else { 0xdeadbeef }).collect();
            let expected: Vec<_> = initial.iter().enumerate().map(|(i, &v)| {
                if i >= words { return v; }
                match i as u32 % 4 {
                    0 => 100_000 + i as u32,
                    1 => v + 2 * (8 * v + 100),
                    2 => v + 3 * (16 * v + 136),
                    _ => v + 4 * (8 * v + 36),
                }
            }).collect();
            let reference = runner::dispatch(&whole, [2, 3, 2], &[], &initial).unwrap();
            assert_eq!(reference, expected, "unsplit {threads}");
            let result = runner::dispatch_split(&chain, [2, 3, 2], &initial).unwrap();
            assert_eq!(result, expected, "split {threads}");
            println!("CHAIN_BRANCH threads={threads} pieces={} scratch_words_per_group={} checked_words={words} guards=16", chain.pieces.len(), chain.dwords_per_group);
        }
    }

}
