use async_trait::async_trait;

use crate::config::ComputerUseConfig;

use super::cache::{SnapshotCache, SnapshotKey};
use super::harness::{ComputerHarness, HarnessRequestContext};
use super::types::{AppInfo, AppSnapshot, AxWalkLimits, IndexedElement};

/// Deterministic harness for Linux CI and unit tests.
pub struct MockHarness {
    config: ComputerUseConfig,
}

impl MockHarness {
    pub fn new(config: ComputerUseConfig) -> Self {
        Self { config }
    }

    fn calculator_snapshot(&self, generation: u64) -> AppSnapshot {
        AppSnapshot {
            generation,
            bundle_id: "com.apple.calculator".to_string(),
            app_name: "Calculator".to_string(),
            pid: 4242,
            window_id: 99,
            window_title: "Calculator".to_string(),
            elements: vec![
                IndexedElement {
                    index: 1,
                    role: "AXButton".to_string(),
                    title: "7".to_string(),
                    enabled: true,
                    bounds: Some(super::types::ElementBounds {
                        x: 10.0,
                        y: 20.0,
                        width: 40.0,
                        height: 40.0,
                    }),
                    subrole: None,
                    interactive: true,
                },
                IndexedElement {
                    index: 2,
                    role: "AXButton".to_string(),
                    title: "+".to_string(),
                    enabled: true,
                    bounds: None,
                    subrole: None,
                    interactive: true,
                },
            ],
            truncated: false,
            png: vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0],
        }
    }
}

#[async_trait]
impl ComputerHarness for MockHarness {
    fn check_permissions(&self) -> Result<(), String> {
        Ok(())
    }

    async fn list_apps(&self) -> Result<Vec<AppInfo>, String> {
        Ok(vec![AppInfo {
            name: "Calculator".to_string(),
            bundle_id: "com.apple.calculator".to_string(),
            pid: 4242,
        }])
    }

    async fn launch_app(&self, app: &str) -> Result<AppInfo, String> {
        if app.eq_ignore_ascii_case("calculator")
            || app.eq_ignore_ascii_case("com.apple.calculator")
        {
            return Ok(AppInfo {
                name: "Calculator".to_string(),
                bundle_id: "com.apple.calculator".to_string(),
                pid: 4242,
            });
        }
        Err(format!(
            "Mock harness can only launch Calculator, not '{app}'"
        ))
    }

    async fn get_app_state(
        &self,
        app: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String> {
        let _limits = AxWalkLimits::from_config(&self.config);
        if !app.eq_ignore_ascii_case("calculator")
            && !app.eq_ignore_ascii_case("com.apple.calculator")
        {
            return Err(format!(
                "Mock harness only supports Calculator, not '{app}'"
            ));
        }
        let key = SnapshotKey {
            task_id: ctx.task_id.clone(),
            session_id: ctx.session_id.clone(),
            bundle_id: "com.apple.calculator".to_string(),
        };
        let snap = self.calculator_snapshot(0);
        Ok(cache.store(key, snap))
    }

    async fn capture_screenshot(&self, app: &str) -> Result<AppSnapshot, String> {
        if !app.eq_ignore_ascii_case("calculator")
            && !app.eq_ignore_ascii_case("com.apple.calculator")
        {
            return Err(format!(
                "Mock harness only supports Calculator, not '{app}'"
            ));
        }
        // Image only: no cache.store, so the element generation is not advanced.
        let mut snap = self.calculator_snapshot(0);
        snap.elements.clear();
        Ok(snap)
    }

    async fn activate_app(
        &self,
        app: &str,
        generation: Option<u64>,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String> {
        let key = snapshot_key(app, ctx)?;
        if let Some(generation) = generation {
            cache.validate_generation(&key, generation)?;
        }
        let mut snap = self.calculator_snapshot(generation.unwrap_or(0));
        snap.window_title = "Calculator (active)".to_string();
        Ok(cache.store(key, snap))
    }

    async fn click(
        &self,
        app: &str,
        generation: u64,
        element_index: Option<u32>,
        _x: Option<f64>,
        _y: Option<f64>,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<(AppSnapshot, Option<u32>, &'static str), String> {
        let key = snapshot_key(app, ctx)?;
        if let Some(index) = element_index {
            let _el = cache.element_by_index(&key, generation, index)?;
        }
        let mut snap = self.calculator_snapshot(generation);
        snap.window_title = "Calculator (clicked)".to_string();
        Ok((cache.store(key, snap), element_index, "mock"))
    }

    async fn type_text(
        &self,
        app: &str,
        generation: u64,
        element_index: Option<u32>,
        _text: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String> {
        let key = snapshot_key(app, ctx)?;
        cache.validate_generation(&key, generation)?;
        if let Some(index) = element_index {
            // Validate the targeted element exists, mirroring the macOS focus step.
            let _ = cache.element_by_index(&key, generation, index)?;
        }
        Ok(cache.store(key, self.calculator_snapshot(generation)))
    }

    async fn press_key(
        &self,
        app: &str,
        generation: u64,
        _key: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String> {
        let key = snapshot_key(app, ctx)?;
        cache.validate_generation(&key, generation)?;
        Ok(cache.store(key, self.calculator_snapshot(generation)))
    }

    async fn scroll(
        &self,
        app: &str,
        generation: u64,
        element_index: Option<u32>,
        _direction: &str,
        _pages: f64,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<(AppSnapshot, Option<u32>), String> {
        let key = snapshot_key(app, ctx)?;
        match element_index {
            Some(index) => {
                let _el = cache.element_by_index(&key, generation, index)?;
            }
            None => {
                cache.validate_generation(&key, generation)?;
            }
        }
        Ok((
            cache.store(key, self.calculator_snapshot(generation)),
            element_index,
        ))
    }

    async fn set_value(
        &self,
        app: &str,
        generation: u64,
        element_index: u32,
        _value: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<(AppSnapshot, u32), String> {
        let key = snapshot_key(app, ctx)?;
        let _el = cache.element_by_index(&key, generation, element_index)?;
        Ok((
            cache.store(key, self.calculator_snapshot(generation)),
            element_index,
        ))
    }
}

fn snapshot_key(app: &str, ctx: &HarnessRequestContext) -> Result<SnapshotKey, String> {
    if !app.eq_ignore_ascii_case("calculator") && !app.eq_ignore_ascii_case("com.apple.calculator")
    {
        return Err(format!(
            "Mock harness only supports Calculator, not '{app}'"
        ));
    }
    Ok(SnapshotKey {
        task_id: ctx.task_id.clone(),
        session_id: ctx.session_id.clone(),
        bundle_id: "com.apple.calculator".to_string(),
    })
}
