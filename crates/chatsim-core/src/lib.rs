//! Terminal-independent logic for `chatsim`, a console chat room for trying
//! increbot without Discord.
//!
//! The room reuses the bot's real chat logic from `increbot::chat` (reply gate,
//! history window, prompt template, merging queue, sanitising), so what the
//! model sees here is what it would see in a Discord channel. Replies come from
//! a [`Backend`]: the real inference thread ([`EngineBackend`]) or a scripted
//! stand-in that needs no model ([`ScriptedBackend`]).
//!
//! [`App`] ties a [`Room`], a backend and the input line together behind plain
//! [`Action`]s, so a front end only maps keys to actions and draws the state.

pub mod app;
pub mod backend;
pub mod command;
pub mod engine;
pub mod input;
pub mod room;
pub mod scripted;

pub use app::{Action, App};
pub use backend::{Backend, BackendEvent, ReplyRequest, ReplyStats};
pub use command::{Command, ReplyMode};
pub use engine::EngineBackend;
pub use input::InputLine;
pub use room::{Activity, ChatMessage, Effect, Kind, Room, RoomConfig};
pub use scripted::{Pacing, ScriptHandle, ScriptedBackend};
