//! Where a loopback service can be reached from outside: a live origin, when
//! something publishes one, then the static origins from config.
//!
//! The services hold an [`EndpointSource`]; whatever publishes them (a
//! tunnel supervisor) holds the matching [`EndpointPublisher`]. Origins are
//! plain strings (`https://host[:port]`), validated by the publisher before
//! they are set.

use std::sync::Arc;

use tokio::sync::watch;

/// The origins a service is reachable at, live origin first.
#[derive(Clone, Debug)]
pub struct EndpointSource {
    live: Option<watch::Receiver<Option<String>>>,
    fallback: Arc<[String]>,
}

/// Sets the live origin of one [`EndpointSource`].
#[derive(Debug)]
pub struct EndpointPublisher {
    tx: watch::Sender<Option<String>>,
}

fn same_origin(a: &str, b: &str) -> bool {
    a.trim_end_matches('/') == b.trim_end_matches('/')
}

impl EndpointSource {
    /// Only these origins, never a live one.
    pub fn fixed(origins: Vec<String>) -> Self {
        Self {
            live: None,
            fallback: origins.into(),
        }
    }

    /// A source whose live origin the returned publisher sets; `fallback`
    /// is served after it, and alone while there is none.
    pub fn live(fallback: Vec<String>) -> (Self, EndpointPublisher) {
        let (tx, rx) = watch::channel(None);
        (
            Self {
                live: Some(rx),
                fallback: fallback.into(),
            },
            EndpointPublisher { tx },
        )
    }

    /// The live origin, if one is published now.
    pub fn live_origin(&self) -> Option<String> {
        self.live.as_ref().and_then(|rx| rx.borrow().clone())
    }

    /// The live origin first, then the fallback, without duplicates.
    pub fn current(&self) -> Vec<String> {
        let mut origins: Vec<String> = Vec::with_capacity(self.fallback.len() + 1);
        for origin in self
            .live_origin()
            .into_iter()
            .chain(self.fallback.iter().cloned())
        {
            if !origins.iter().any(|known| same_origin(known, &origin)) {
                origins.push(origin);
            }
        }
        origins
    }
}

impl EndpointPublisher {
    /// Publish `origin` as the live one; `None` while it is down.
    pub fn set(&self, origin: Option<String>) {
        self.tx.send_replace(origin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_serves_config_origins() {
        let source = EndpointSource::fixed(vec!["https://a.example.com".into()]);
        assert_eq!(source.current(), vec!["https://a.example.com"]);
        assert_eq!(source.live_origin(), None);
    }

    #[test]
    fn live_origin_comes_first_and_is_deduplicated() {
        let (source, publisher) = EndpointSource::live(vec![
            "https://a.example.com".into(),
            "https://b.example.com".into(),
        ]);
        assert_eq!(source.current().len(), 2);
        publisher.set(Some("https://b.example.com/".into()));
        assert_eq!(
            source.current(),
            vec!["https://b.example.com/", "https://a.example.com"]
        );
        let clone = source.clone();
        publisher.set(None);
        assert_eq!(
            clone.current(),
            vec!["https://a.example.com", "https://b.example.com"]
        );
    }

    #[test]
    fn source_outlives_its_publisher() {
        let (source, publisher) = EndpointSource::live(vec![]);
        publisher.set(Some("https://c.example.com".into()));
        drop(publisher);
        assert_eq!(source.current(), vec!["https://c.example.com"]);
    }
}
