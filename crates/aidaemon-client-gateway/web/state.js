// What the client knows, built only from server frames: sessions, each
// session's transcript, the approvals waiting, the work running, and the
// last `seq` applied. No DOM here: the same reducer runs in the browser, in
// the desktop shell and under Node for the shared fixtures
// (crates/aidaemon-client-core/tests/fixtures/reducer), which hold it to the
// Rust reducer in aidaemon-client-core. Its JSON shape is the Rust `State`'s.
export function newState() {
    return { last_seq: null, sessions: {}, order: [], approvals: {}, tasks: {}, notice: null };
}
/** A turn status in a few words. */
export function statusText(status) {
    switch (status.kind) {
        case "thinking":
            return "Thinking...";
        case "tool_started":
            return status.summary ? `${status.name}: ${status.summary}...` : `${status.name}...`;
        case "tool_progress":
            return `${status.name}: ${[...status.chunk].slice(0, 80).join("")}`;
        case "tool_completed":
            return `✓ ${status.name} ${status.summary ?? ""}`;
        case "plan_step":
            return `Step ${status.step + 1} of ${status.total}: ${status.description}`;
        case "checklist":
            return "Updated the plan";
        case "note":
            return status.text;
    }
    return "Working...";
}
function session(state, id) {
    let view = state.sessions[id];
    if (!view) {
        state.order.unshift(id);
        view = state.sessions[id] = { summary: null, lines: [], live: null };
    }
    return view;
}
/** Move `id` to the top of the list. */
function touch(state, id) {
    session(state, id);
    state.order = state.order.filter((other) => other !== id);
    state.order.unshift(id);
}
function reset(state, seq, snapshot) {
    const transcripts = state.sessions;
    state.sessions = {};
    state.order = [];
    state.approvals = {};
    state.tasks = {};
    for (const summary of snapshot.sessions) {
        const old = transcripts[summary.id];
        state.sessions[summary.id] = { summary, lines: old ? old.lines : [], live: null };
        state.order.push(summary.id);
    }
    for (const approval of snapshot.pending_approvals)
        state.approvals[approval.id] = approval;
    for (const task of snapshot.running_tasks)
        state.tasks[task.id] = task;
    state.last_seq = seq;
}
/**
 * Apply one frame from the gateway. Durable events at or below the last
 * `seq` applied are ignored, so a replay never doubles a line.
 */
export function apply(state, frame) {
    switch (frame.type) {
        case "snapshot":
            reset(state, frame.seq, frame.state);
            return;
        case "event": {
            const seq = frame.seq;
            if (seq != null) {
                if (state.last_seq != null && seq <= state.last_seq)
                    return;
                state.last_seq = seq;
            }
            applyEvent(state, frame.session_id ?? null, frame.event);
            return;
        }
        case "ack":
            applyResult(state, frame.result);
            return;
        case "error":
            state.notice = frame.error.message;
            return;
    }
}
function applyResult(state, result) {
    switch (result.type) {
        case "snapshot":
            reset(state, result.seq, result.state);
            return;
        case "sessions":
            for (const summary of result.sessions)
                session(state, summary.id).summary = summary;
            return;
        case "history":
            setHistory(state, result.session_id, result.messages);
            return;
        case "stopped":
            state.notice = result.report;
            return;
        case "reply":
            state.notice = result.text;
            return;
    }
}
function setHistory(state, sessionId, messages) {
    const view = session(state, sessionId);
    // History is older than anything live: it goes first.
    const lines = messages.map((m) => m.role === "user" ? { kind: "user", text: m.text, origin: "" } : { kind: "assistant", text: m.text });
    view.lines = lines.concat(view.lines);
}
function liveTurn(state, turnId) {
    return Object.values(state.sessions).find((view) => view.live?.turn_id === turnId);
}
function applyEvent(state, sessionId, event) {
    switch (event.type) {
        case "session_created":
        case "session_updated":
            touch(state, event.session.id);
            session(state, event.session.id).summary = event.session;
            return;
        case "user_message":
            touch(state, event.session_id);
            session(state, event.session_id).lines.push({ kind: "user", text: event.text, origin: event.origin });
            return;
        case "assistant_message":
            session(state, event.session_id).lines.push({ kind: "assistant", text: event.text });
            return;
        case "turn_started":
            session(state, event.session_id).live = { turn_id: event.turn_id, streamed: "", status: null };
            return;
        case "turn_delta": {
            const view = liveTurn(state, event.turn_id);
            if (view?.live)
                view.live.streamed += event.text;
            return;
        }
        case "turn_status": {
            const view = liveTurn(state, event.turn_id);
            if (view?.live)
                view.live.status = statusText(event.status);
            return;
        }
        case "user_action_required":
            if (sessionId != null)
                session(state, sessionId).lines.push({ kind: "note", text: event.message });
            return;
        case "turn_completed": {
            const view = liveTurn(state, event.turn_id);
            if (!view) {
                // Its start was before this client's window.
                if (sessionId != null && event.text) {
                    session(state, sessionId).lines.push({ kind: "assistant", text: event.text });
                }
                return;
            }
            view.live = null;
            if (event.text)
                view.lines.push({ kind: "assistant", text: event.text });
            if (event.state === "cancelled")
                view.lines.push({ kind: "note", text: "Stopped." });
            if (event.state === "failed")
                view.lines.push({ kind: "note", text: `Failed: ${event.error ?? ""}` });
            return;
        }
        case "approval_requested":
            state.approvals[event.approval.id] = event.approval;
            return;
        case "approval_resolved":
            delete state.approvals[event.approval_id];
            return;
        case "task_updated":
            if (event.task.state === "running" || event.task.state === "cancelling") {
                state.tasks[event.task.id] = event.task;
            }
            else {
                delete state.tasks[event.task.id];
            }
            return;
    }
}
/** The oldest approval waiting, which the card shows. */
export function nextApproval(state) {
    return Object.values(state.approvals).sort((a, b) => a.requested_at_ms - b.requested_at_ms)[0];
}
/** Running tasks, oldest first. */
export function running(state) {
    return Object.values(state.tasks).sort((a, b) => a.started_at_ms - b.started_at_ms);
}
function short(id) {
    const chars = [...id];
    return chars.length > 24 ? `${chars.slice(0, 23).join("")}…` : id;
}
export function title(state, id) {
    const summary = state.sessions[id]?.summary;
    if (summary?.title)
        return summary.title;
    if (summary)
        return `${short(id)} (${summary.channel})`;
    return short(id);
}
