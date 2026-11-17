use std::sync::Arc;
use std::time::Duration;

use proptest::prelude::*;
use tokio::time::Instant;

use super::*;
use crate::testing::MockSurface;
use crate::types::{InputEvent, PointerAction, PointerButton, StreamParams};

fn setup(config: SurfacesConfig) -> (SurfaceRegistry, MockSurface) {
    let registry = SurfaceRegistry::new(config);
    let mock = MockSurface::new();
    mock.set_notifier(registry.notifier());
    registry.add_backend(Arc::new(mock.clone()));
    (registry, mock)
}

fn tap() -> InputEvent {
    InputEvent::Pointer {
        action: PointerAction::Down,
        x: 0.5,
        y: 0.5,
        button: PointerButton::Left,
    }
}

fn text(t: &str) -> InputEvent {
    InputEvent::Text { text: t.into() }
}

#[tokio::test]
async fn refresh_publishes_new_changed_and_closed_surfaces() {
    let (registry, mock) = setup(SurfacesConfig::default());
    let mut changes = registry.subscribe();
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    assert!(matches!(changes.try_recv(), Ok(SurfaceChange::Updated(info)) if info.id == id));
    registry.refresh().await;
    assert!(
        changes.try_recv().is_err(),
        "an unchanged surface is not republished"
    );
    mock.set_title(&id, "Sign in");
    registry.refresh().await;
    assert!(
        matches!(changes.try_recv(), Ok(SurfaceChange::Updated(info)) if info.title == "Sign in")
    );
    mock.remove(&id);
    registry.refresh().await;
    assert_eq!(changes.try_recv(), Ok(SurfaceChange::Closed(id)));
}

#[tokio::test]
async fn the_agent_finds_a_surface_by_its_tab_or_the_active_one() {
    let (registry, mock) = setup(SurfacesConfig::default());
    let first = mock.add("s1", "tab-1");
    let second = mock.add("s1", "tab-2");
    mock.add("s2", "tab-1");
    let by_ref = registry.find_for_agent("s1", Some("tab-1")).await.unwrap();
    assert_eq!(by_ref.id, first);
    let active = registry.find_for_agent("s1", None).await.unwrap();
    assert_eq!(active.id, second);
    assert!(registry.find_for_agent("s1", Some("tab-9")).await.is_none());
    assert!(registry.find_for_agent("s3", None).await.is_none());
}

#[tokio::test]
async fn viewers_share_one_stream_that_stops_with_the_last() {
    let (registry, mock) = setup(SurfacesConfig::default());
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    let mut a = registry
        .open_live(&id, "phone", LiveMode::View, StreamParams::default())
        .await
        .unwrap();
    let mut b = registry
        .open_live(&id, "laptop", LiveMode::View, StreamParams::default())
        .await
        .unwrap();
    assert_eq!(mock.streams_started(), 1);
    assert!(mock.push_frame(&id, b"one"));
    assert!(mock.push_frame(&id, b"two"));
    // Latest wins: a slow viewer sees only the newest frame.
    a.frames.changed().await.unwrap();
    let frame = a.frames.borrow_and_update().clone().unwrap();
    assert_eq!(&*frame.data, b"two");
    assert_eq!(frame.seq, 2);
    b.frames.changed().await.unwrap();
    registry.close_live(&a.live_id);
    assert_eq!(mock.streams_stopped(), 0);
    registry.close_live(&b.live_id);
    assert_eq!(mock.streams_stopped(), 1);
    assert!(*a.ended.borrow());
}

/// A view whose own receivers are dropped (its socket subscribes later)
/// keeps its stream watched, so the backend never stops sending.
#[tokio::test]
async fn a_stream_stays_watched_until_its_view_closes() {
    let (registry, mock) = setup(SurfacesConfig::default());
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    let live = registry
        .open_live(&id, "phone", LiveMode::View, StreamParams::default())
        .await
        .unwrap();
    let live_id = live.live_id.clone();
    drop(live);
    assert!(
        mock.push_frame(&id, b"one"),
        "still watched before the socket subscribes"
    );
    let mut later = registry.subscribe_live(&live_id).unwrap();
    assert_eq!(
        &*later.frames.borrow_and_update().clone().unwrap().data,
        b"one"
    );
    registry.close_live(&live_id);
    assert_eq!(mock.streams_stopped(), 1);
}

#[tokio::test]
async fn watching_needs_no_control_and_sends_no_input() {
    let (registry, mock) = setup(SurfacesConfig::default());
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    let live = registry
        .open_live(&id, "phone", LiveMode::View, StreamParams::default())
        .await
        .unwrap();
    assert_eq!(live.surface.controller, Controller::Agent);
    let refused = registry.input(&live.live_id, &tap()).await;
    assert!(matches!(refused, Err(SurfaceError::Refused(_))));
    assert!(mock.inputs().is_empty());
}

#[tokio::test]
async fn without_control_anytime_only_a_handoff_allows_control() {
    let config = SurfacesConfig {
        control_anytime: false,
        ..SurfacesConfig::default()
    };
    let (registry, mock) = setup(config);
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    let refused = registry
        .open_live(&id, "phone", LiveMode::Control, StreamParams::default())
        .await;
    assert!(matches!(refused, Err(SurfaceError::Refused(_))));
    registry.begin_handoff(&id).unwrap();
    let live = registry
        .open_live(&id, "phone", LiveMode::Control, StreamParams::default())
        .await
        .unwrap();
    assert_eq!(
        live.surface.controller,
        Controller::Owner {
            device_id: "phone".into()
        }
    );
    registry.input(&live.live_id, &text("code")).await.unwrap();
    assert_eq!(mock.inputs().len(), 1);
    registry.end_handoff(&id);
    let info = registry.get(&id).unwrap();
    assert_eq!(info.controller, Controller::Agent);
    assert!(!info.handoff_pending);
    assert!(registry.input(&live.live_id, &text("x")).await.is_err());
}

#[tokio::test]
async fn control_waits_for_the_agents_running_action_and_holds_the_next() {
    let (registry, mock) = setup(SurfacesConfig::default());
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    let agent_lock = mock.agent_lock(&id);
    let running = agent_lock.clone().lock_owned().await;
    let take = {
        let registry = registry.clone();
        let id = id.clone();
        tokio::spawn(async move { registry.take_control(&id, "phone").await })
    };
    tokio::task::yield_now().await;
    assert_eq!(registry.get(&id).unwrap().controller, Controller::Agent);
    drop(running);
    take.await.unwrap().unwrap();
    assert!(
        agent_lock.try_lock().is_err(),
        "the agent's next action waits"
    );
    registry.release_control(&id, "phone").unwrap();
    assert!(agent_lock.try_lock().is_ok());
}

#[tokio::test]
async fn a_page_takes_the_controlling_phones_room_and_is_back_before_the_agent_acts() {
    let (registry, mock) = setup(SurfacesConfig::default());
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    let room = crate::types::Size { w: 390, h: 560 };
    let phone = StreamParams {
        fit: Some(room),
        ..StreamParams::default()
    };
    // Watching lays nothing out; a laptop with no room changes nothing.
    let _watch = registry
        .open_live(&id, "phone", LiveMode::View, phone)
        .await
        .unwrap();
    let _laptop = registry
        .open_live(&id, "laptop", LiveMode::View, StreamParams::default())
        .await
        .unwrap();
    registry.take_control(&id, "laptop").await.unwrap();
    assert!(mock.fits().is_empty());
    // The phone takes over: the page takes its room.
    registry.take_control(&id, "phone").await.unwrap();
    assert_eq!(mock.fits(), vec![(Some(room), true)]);
    // Given back: its own size again while the agent still waits, then the
    // agent may act.
    let agent_lock = mock.agent_lock(&id);
    registry.release_control(&id, "phone").unwrap();
    let _acting = tokio::time::timeout(Duration::from_secs(5), agent_lock.lock())
        .await
        .expect("the agent acts again");
    assert_eq!(mock.fits(), vec![(Some(room), true), (None, true)]);
}

#[tokio::test]
async fn only_the_holding_device_types_and_the_newest_device_wins() {
    let (registry, mock) = setup(SurfacesConfig::default());
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    let phone = registry
        .open_live(&id, "phone", LiveMode::Control, StreamParams::default())
        .await
        .unwrap();
    let laptop = registry
        .open_live(&id, "laptop", LiveMode::View, StreamParams::default())
        .await
        .unwrap();
    assert!(registry.input(&laptop.live_id, &tap()).await.is_err());
    registry.take_control(&id, "laptop").await.unwrap();
    assert!(registry.input(&phone.live_id, &tap()).await.is_err());
    registry.input(&laptop.live_id, &tap()).await.unwrap();
    // Releasing from a device that does not hold it changes nothing.
    registry.release_control(&id, "phone").unwrap();
    assert_eq!(
        registry.get(&id).unwrap().controller.owner_device(),
        Some("laptop")
    );
}

#[tokio::test(start_paused = true)]
async fn control_goes_back_when_the_device_leaves_or_goes_quiet() {
    let config = SurfacesConfig {
        disconnect_grace: Duration::from_secs(30),
        idle_release: Duration::from_secs(600),
        ..SurfacesConfig::default()
    };
    let (registry, mock) = setup(config);
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    let live = registry
        .open_live(&id, "phone", LiveMode::Control, StreamParams::default())
        .await
        .unwrap();
    registry.close_live(&live.live_id);
    registry.reap(Instant::now() + Duration::from_secs(10));
    assert!(
        registry
            .get(&id)
            .unwrap()
            .controller
            .owner_device()
            .is_some()
    );
    registry.reap(Instant::now() + Duration::from_secs(31));
    assert_eq!(registry.get(&id).unwrap().controller, Controller::Agent);

    let live = registry
        .open_live(&id, "phone", LiveMode::Control, StreamParams::default())
        .await
        .unwrap();
    registry.reap(Instant::now() + Duration::from_secs(599));
    assert!(
        registry
            .get(&id)
            .unwrap()
            .controller
            .owner_device()
            .is_some()
    );
    registry.reap(Instant::now() + Duration::from_secs(601));
    assert_eq!(registry.get(&id).unwrap().controller, Controller::Agent);
    drop(live);
}

#[tokio::test]
async fn a_closed_surface_ends_its_live_sessions() {
    let (registry, mock) = setup(SurfacesConfig::default());
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    let live = registry
        .open_live(&id, "phone", LiveMode::View, StreamParams::default())
        .await
        .unwrap();
    mock.remove(&id);
    registry.refresh().await;
    assert!(*live.ended.borrow());
    assert_eq!(registry.live_count(), 0);
}

#[tokio::test]
async fn limits_and_switches_hold() {
    let config = SurfacesConfig {
        max_live_sessions: 1,
        ..SurfacesConfig::default()
    };
    let (registry, mock) = setup(config.clone());
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    let _one = registry
        .open_live(&id, "phone", LiveMode::Control, StreamParams::default())
        .await
        .unwrap();
    let two = registry
        .open_live(&id, "laptop", LiveMode::View, StreamParams::default())
        .await;
    assert!(matches!(two, Err(SurfaceError::TooMany)));
    registry.set_config(SurfacesConfig {
        allow_control: false,
        ..config
    });
    assert_eq!(registry.get(&id).unwrap().controller, Controller::Agent);
    registry.set_config(SurfacesConfig {
        enabled: false,
        ..SurfacesConfig::default()
    });
    assert_eq!(registry.live_count(), 0);
}

#[tokio::test]
async fn bad_input_is_refused_before_the_backend() {
    let (registry, mock) = setup(SurfacesConfig::default());
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    let live = registry
        .open_live(&id, "phone", LiveMode::Control, StreamParams::default())
        .await
        .unwrap();
    let bad = InputEvent::Pointer {
        action: PointerAction::Down,
        x: 3.0,
        y: 0.1,
        button: PointerButton::Left,
    };
    assert!(matches!(
        registry.input(&live.live_id, &bad).await,
        Err(SurfaceError::BadInput(_))
    ));
    assert!(mock.inputs().is_empty());
}

#[tokio::test]
async fn a_handoff_may_reopen_the_surface() {
    let (registry, mock) = setup(SurfacesConfig::default());
    let id = mock.add("s1", "tab-1");
    registry.refresh().await;
    mock.reopen_on_handoff();
    let prep = registry.prepare_handoff(&id).await.unwrap();
    assert_ne!(prep.id, id);
    assert!(registry.get(&prep.id).is_some());
    assert!(registry.get(&id).is_none());
}

#[derive(Debug, Clone)]
enum Op {
    Take(u8),
    Release(u8),
    Open(u8),
    CloseAll,
    Reap(u16),
    BeginHandoff,
    EndHandoff,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0u8..3).prop_map(Op::Take),
        (0u8..3).prop_map(Op::Release),
        (0u8..3).prop_map(Op::Open),
        Just(Op::CloseAll),
        (0u16..1200).prop_map(Op::Reap),
        Just(Op::BeginHandoff),
        Just(Op::EndHandoff),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Whatever happens, one controller at a time; the agent's lock is held
    /// exactly while an owner controls; a released or reaped lease frees it.
    #[test]
    fn the_lease_stays_consistent(ops in proptest::collection::vec(op(), 1..40)) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            let (registry, mock) = setup(SurfacesConfig::default());
            let id = mock.add("s1", "tab-1");
            registry.refresh().await;
            let lock = mock.agent_lock(&id);
            let mut lives: Vec<String> = Vec::new();
            let base = Instant::now();
            for op in ops {
                match op {
                    Op::Take(d) => {
                        registry.take_control(&id, &format!("d{d}")).await.unwrap();
                    }
                    Op::Release(d) => {
                        registry.release_control(&id, &format!("d{d}")).unwrap();
                    }
                    Op::Open(d) => {
                        let live = registry
                            .open_live(&id, &format!("d{d}"), LiveMode::View, StreamParams::default())
                            .await;
                        if let Ok(live) = live {
                            lives.push(live.live_id);
                        }
                    }
                    Op::CloseAll => {
                        for live in lives.drain(..) {
                            registry.close_live(&live);
                        }
                    }
                    Op::Reap(secs) => registry.reap(base + Duration::from_secs(u64::from(secs))),
                    Op::BeginHandoff => registry.begin_handoff(&id).unwrap(),
                    Op::EndHandoff => registry.end_handoff(&id),
                }
                let controller = registry.get(&id).unwrap().controller;
                let agent_free = lock.try_lock().is_ok();
                prop_assert_eq!(controller == Controller::Agent, agent_free);
            }
            Ok(())
        })?;
    }
}
