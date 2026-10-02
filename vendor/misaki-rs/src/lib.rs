pub mod data;
pub mod fallback;
pub mod g2p;
pub mod language;
pub mod languages;
pub mod lexicon;
pub mod tagger;
pub mod token;

#[cfg(feature = "espeak")]
pub use fallback::EspeakFallback;
pub use fallback::Fallback;
pub use g2p::G2P;
pub use language::Language;
pub use lexicon::Lexicon;
pub use token::MToken;
