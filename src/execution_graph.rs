//! Small, typed graph primitives for durable execution state.
//!
//! This module deliberately models lifecycle relationships rather than natural
//! language. Completion, plans, task dependencies, dialogue obligations, and
//! recovery can therefore share the same invariants without sharing phrase
//! classifiers.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

const MAX_GRAPH_NODES: usize = 512;
const MAX_GRAPH_EDGES: usize = 2_048;
const MAX_NODE_ID_CHARS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ExecutionNodeKind {
    Request,
    Obligation,
    PlanStep,
    Task,
    Receipt,
    Verification,
    HumanInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecutionNodeState {
    Pending,
    Running,
    Satisfied,
    Failed,
    Blocked,
    Invalidated,
    Superseded,
}

impl ExecutionNodeState {
    pub(crate) fn satisfies_dependency(self) -> bool {
        matches!(self, Self::Satisfied | Self::Superseded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ExecutionEdgeKind {
    /// `from` cannot start until `to` is satisfied.
    DependsOn,
    /// `from` cannot complete until `to` is satisfied.
    Requires,
    /// Evidence node `from` proves obligation node `to`.
    Satisfies,
    Invalidates,
    AwaitsInput,
}

impl ExecutionEdgeKind {
    fn is_precedence(self) -> bool {
        matches!(self, Self::DependsOn | Self::Requires | Self::AwaitsInput)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExecutionNode {
    pub id: String,
    pub kind: ExecutionNodeKind,
    pub state: ExecutionNodeState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExecutionEdge {
    pub from: String,
    pub to: String,
    pub kind: ExecutionEdgeKind,
    pub evidence_id: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ExecutionGraph {
    nodes: BTreeMap<String, ExecutionNode>,
    edges: Vec<ExecutionEdge>,
}

/// Persistence-independent projection used to build a task dependency graph.
pub(crate) struct ExecutionTaskNode {
    pub id: String,
    pub status: String,
    pub dependency_ids: Vec<String>,
}

impl ExecutionGraph {
    pub(crate) fn from_task_nodes(tasks: &[ExecutionTaskNode]) -> Result<Self, String> {
        let mut graph = Self::default();
        for task in tasks {
            if graph.nodes.contains_key(&task.id) {
                return Err(format!("task graph contains duplicate task ID {}", task.id));
            }
            graph.add_node(
                task.id.clone(),
                ExecutionNodeKind::Task,
                task_execution_state(&task.status),
            )?;
        }
        for task in tasks {
            for dependency_id in &task.dependency_ids {
                if !graph.nodes.contains_key(dependency_id) {
                    return Err(format!(
                        "task {} depends on missing task {}",
                        task.id, dependency_id
                    ));
                }
                graph.add_edge(&task.id, dependency_id, ExecutionEdgeKind::DependsOn, None)?;
            }
        }
        Ok(graph)
    }

    pub(crate) fn add_node(
        &mut self,
        id: impl Into<String>,
        kind: ExecutionNodeKind,
        state: ExecutionNodeState,
    ) -> Result<(), String> {
        let id = id.into();
        validate_id(&id)?;
        if let Some(existing) = self.nodes.get(&id) {
            if existing.kind != kind {
                return Err(format!(
                    "execution node {id} already exists as {:?}, not {:?}",
                    existing.kind, kind
                ));
            }
            return Ok(());
        }
        if self.nodes.len() >= MAX_GRAPH_NODES {
            return Err(format!(
                "execution graph exceeds the {MAX_GRAPH_NODES}-node safety limit"
            ));
        }
        self.nodes
            .insert(id.clone(), ExecutionNode { id, kind, state });
        Ok(())
    }

    pub(crate) fn set_state(&mut self, id: &str, state: ExecutionNodeState) -> Result<(), String> {
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| format!("execution node {id} does not exist"))?;
        node.state = state;
        Ok(())
    }

    pub(crate) fn state(&self, id: &str) -> Option<ExecutionNodeState> {
        self.nodes.get(id).map(|node| node.state)
    }

    pub(crate) fn node_kind(&self, id: &str) -> Option<ExecutionNodeKind> {
        self.nodes.get(id).map(|node| node.kind)
    }

    pub(crate) fn add_edge(
        &mut self,
        from: &str,
        to: &str,
        kind: ExecutionEdgeKind,
        evidence_id: Option<String>,
    ) -> Result<(), String> {
        if !self.nodes.contains_key(from) {
            return Err(format!("execution edge source {from} does not exist"));
        }
        if !self.nodes.contains_key(to) {
            return Err(format!("execution edge target {to} does not exist"));
        }
        if from == to && kind.is_precedence() {
            return Err(format!("execution node {from} cannot depend on itself"));
        }
        if self.edges.iter().any(|edge| {
            edge.from == from
                && edge.to == to
                && edge.kind == kind
                && edge.evidence_id == evidence_id
        }) {
            return Ok(());
        }
        if self.edges.len() >= MAX_GRAPH_EDGES {
            return Err(format!(
                "execution graph exceeds the {MAX_GRAPH_EDGES}-edge safety limit"
            ));
        }

        self.edges.push(ExecutionEdge {
            from: from.to_string(),
            to: to.to_string(),
            kind,
            evidence_id,
        });
        if kind.is_precedence() && self.has_precedence_cycle() {
            self.edges.pop();
            return Err(format!(
                "adding {kind:?} edge {from} -> {to} would create a lifecycle cycle"
            ));
        }
        Ok(())
    }

    /// Returns the unresolved nodes that `id` directly depends on.
    pub(crate) fn unresolved_dependencies(&self, id: &str) -> Vec<&ExecutionNode> {
        self.edges
            .iter()
            .filter(|edge| edge.from == id && edge.kind == ExecutionEdgeKind::DependsOn)
            .filter_map(|edge| self.nodes.get(&edge.to))
            .filter(|node| !node.state.satisfies_dependency())
            .collect()
    }

    pub(crate) fn dependencies_satisfied(&self, id: &str) -> bool {
        self.nodes.contains_key(id) && self.unresolved_dependencies(id).is_empty()
    }

    pub(crate) fn requirements_satisfied(&self, id: &str) -> bool {
        self.nodes.contains_key(id)
            && self
                .edges
                .iter()
                .filter(|edge| edge.from == id && edge.kind == ExecutionEdgeKind::Requires)
                .all(|edge| {
                    self.nodes
                        .get(&edge.to)
                        .is_some_and(|node| node.state.satisfies_dependency())
                })
    }

    #[cfg(test)]
    pub(crate) fn ready_nodes(&self, kind: ExecutionNodeKind) -> Vec<&ExecutionNode> {
        self.nodes
            .values()
            .filter(|node| {
                node.kind == kind
                    && matches!(node.state, ExecutionNodeState::Pending)
                    && self.dependencies_satisfied(&node.id)
            })
            .collect()
    }

    /// Record typed evidence for an obligation and mark the obligation
    /// satisfied. Only receipt, verification, human-input, artifact, or outcome
    /// nodes are accepted as proof.
    pub(crate) fn satisfy_with_evidence(
        &mut self,
        obligation_id: &str,
        evidence_id: &str,
        receipt_id: Option<String>,
    ) -> Result<(), String> {
        let obligation_kind = self
            .node_kind(obligation_id)
            .ok_or_else(|| format!("obligation node {obligation_id} does not exist"))?;
        if obligation_kind != ExecutionNodeKind::Obligation {
            return Err(format!(
                "execution node {obligation_id} is not an obligation"
            ));
        }
        let evidence_kind = self
            .node_kind(evidence_id)
            .ok_or_else(|| format!("evidence node {evidence_id} does not exist"))?;
        if !matches!(
            evidence_kind,
            ExecutionNodeKind::Receipt
                | ExecutionNodeKind::Verification
                | ExecutionNodeKind::HumanInput
        ) {
            return Err(format!(
                "execution node {evidence_id} ({evidence_kind:?}) cannot prove an obligation"
            ));
        }
        self.add_edge(
            evidence_id,
            obligation_id,
            ExecutionEdgeKind::Satisfies,
            receipt_id,
        )?;
        self.set_state(obligation_id, ExecutionNodeState::Satisfied)
    }

    /// Exact obligations closed by one durable tool receipt. The receipt ID is
    /// stored on `Satisfies` edges even when the graph evidence node is a
    /// verification wrapper, so callers never infer proof from prose or from
    /// mere task-local tool activity.
    pub(crate) fn obligations_satisfied_by_receipt(&self, receipt_id: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|edge| {
                edge.kind == ExecutionEdgeKind::Satisfies
                    && edge.evidence_id.as_deref() == Some(receipt_id)
            })
            .map(|edge| edge.to.clone())
            .collect()
    }

    /// Stable receipt identities currently carrying proof edges. This bounded
    /// projection is used by finalization telemetry so two otherwise identical
    /// completion decisions can be compared without serializing the graph or
    /// any tool output.
    pub(crate) fn satisfying_receipt_ids(&self) -> Vec<String> {
        self.edges
            .iter()
            .filter(|edge| edge.kind == ExecutionEdgeKind::Satisfies)
            .filter_map(|edge| edge.evidence_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .take(MAX_GRAPH_NODES)
            .collect()
    }

    pub(crate) fn invalidate(
        &mut self,
        invalidator_id: &str,
        target_id: &str,
    ) -> Result<(), String> {
        self.add_edge(
            invalidator_id,
            target_id,
            ExecutionEdgeKind::Invalidates,
            None,
        )?;
        self.set_state(target_id, ExecutionNodeState::Invalidated)
    }

    pub(crate) fn has_precedence_cycle(&self) -> bool {
        let mut in_degree: BTreeMap<&str, usize> =
            self.nodes.keys().map(|id| (id.as_str(), 0_usize)).collect();
        let mut dependents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for edge in self.edges.iter().filter(|edge| edge.kind.is_precedence()) {
            // `from -> to` means from depends on to, so topological direction
            // is target prerequisite -> source dependent.
            *in_degree.entry(edge.from.as_str()).or_default() += 1;
            dependents
                .entry(edge.to.as_str())
                .or_default()
                .push(edge.from.as_str());
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
            .collect();
        let mut visited = BTreeSet::new();
        while let Some(id) = queue.pop_front() {
            if !visited.insert(id) {
                continue;
            }
            for dependent in dependents.get(id).into_iter().flatten() {
                if let Some(degree) = in_degree.get_mut(dependent) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        queue.push_back(dependent);
                    }
                }
            }
        }
        visited.len() != self.nodes.len()
    }

    #[cfg(test)]
    pub(crate) fn edges(&self) -> &[ExecutionEdge] {
        &self.edges
    }
}

fn task_execution_state(status: &str) -> ExecutionNodeState {
    match status {
        "completed" | "skipped" => ExecutionNodeState::Satisfied,
        "superseded" => ExecutionNodeState::Superseded,
        "claimed" | "running" => ExecutionNodeState::Running,
        "failed" | "cancelled" | "abandoned" | "interrupted" => ExecutionNodeState::Failed,
        "blocked" => ExecutionNodeState::Blocked,
        _ => ExecutionNodeState::Pending,
    }
}

fn validate_id(id: &str) -> Result<(), String> {
    if id.trim().is_empty() {
        return Err("execution node ID cannot be empty".to_string());
    }
    if id.chars().count() > MAX_NODE_ID_CHARS {
        return Err(format!(
            "execution node ID exceeds {MAX_NODE_ID_CHARS} characters"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_task(graph: &mut ExecutionGraph, id: &str, state: ExecutionNodeState) {
        graph.add_node(id, ExecutionNodeKind::Task, state).unwrap();
    }

    #[test]
    fn join_is_ready_only_after_every_dependency_is_satisfied() {
        let mut graph = ExecutionGraph::default();
        add_task(&mut graph, "left", ExecutionNodeState::Satisfied);
        add_task(&mut graph, "right", ExecutionNodeState::Running);
        add_task(&mut graph, "join", ExecutionNodeState::Pending);
        graph
            .add_edge("join", "left", ExecutionEdgeKind::DependsOn, None)
            .unwrap();
        graph
            .add_edge("join", "right", ExecutionEdgeKind::DependsOn, None)
            .unwrap();

        assert!(!graph.dependencies_satisfied("join"));
        graph
            .set_state("right", ExecutionNodeState::Satisfied)
            .unwrap();
        assert_eq!(
            graph
                .ready_nodes(ExecutionNodeKind::Task)
                .iter()
                .map(|node| node.id.as_str())
                .collect::<Vec<_>>(),
            vec!["join"]
        );
    }

    #[test]
    fn precedence_cycles_are_rejected_without_mutating_graph() {
        let mut graph = ExecutionGraph::default();
        add_task(&mut graph, "a", ExecutionNodeState::Pending);
        add_task(&mut graph, "b", ExecutionNodeState::Pending);
        graph
            .add_edge("a", "b", ExecutionEdgeKind::DependsOn, None)
            .unwrap();
        let error = graph
            .add_edge("b", "a", ExecutionEdgeKind::DependsOn, None)
            .unwrap_err();
        assert!(error.contains("cycle"));
        assert_eq!(graph.edges().len(), 1);
    }

    #[test]
    fn typed_evidence_satisfies_and_invalidation_reopens_obligation() {
        let mut graph = ExecutionGraph::default();
        graph
            .add_node(
                "write-obligation",
                ExecutionNodeKind::Obligation,
                ExecutionNodeState::Pending,
            )
            .unwrap();
        graph
            .add_node(
                "receipt-1",
                ExecutionNodeKind::Receipt,
                ExecutionNodeState::Satisfied,
            )
            .unwrap();
        graph
            .satisfy_with_evidence(
                "write-obligation",
                "receipt-1",
                Some("tool-call-1".to_string()),
            )
            .unwrap();
        assert_eq!(
            graph.state("write-obligation"),
            Some(ExecutionNodeState::Satisfied)
        );

        graph
            .add_node(
                "workspace-change-2",
                ExecutionNodeKind::Receipt,
                ExecutionNodeState::Satisfied,
            )
            .unwrap();
        graph
            .invalidate("workspace-change-2", "write-obligation")
            .unwrap();
        assert_eq!(
            graph.state("write-obligation"),
            Some(ExecutionNodeState::Invalidated)
        );
    }
}
