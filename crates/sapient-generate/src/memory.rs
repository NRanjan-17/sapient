// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Process memory readings: current footprint, peak footprint and how much
//! more the process may allocate.
//!
//! On Apple platforms the footprint is `phys_footprint` from `task_vm_info`,
//! the number iOS compares against an app's memory limit (Xcode's memory
//! gauge shows the same value). It counts dirty and compressed memory, GPU
//! allocations on unified memory included, but not clean memory-mapped file
//! pages, which is why mmap'd weights stay under the limit. Linux and Android
//! report resident set size from `/proc/self/status`.

/// Current memory footprint of this process in bytes, if the platform reports it.
pub fn footprint_bytes() -> Option<u64> {
    imp::footprint_bytes()
}

/// Highest footprint this process has reached, in bytes, if known.
pub fn peak_footprint_bytes() -> Option<u64> {
    imp::peak_footprint_bytes()
}

/// How many more bytes this process can allocate before the OS steps in:
/// the remaining per-app allowance on iOS (`os_proc_available_memory`),
/// available system memory elsewhere. `None` when unknown.
pub fn available_bytes() -> Option<u64> {
    imp::available_bytes()
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod imp {
    use std::mem::{offset_of, size_of};

    /// XNU `struct task_vm_info` (osfmk/mach/task_info.h, `#pragma pack(4)`),
    /// up to revision 4. The kernel fills as much as it supports and reports
    /// how much through the count, so every field is read only if covered.
    #[repr(C, packed(4))]
    #[derive(Clone, Copy)]
    #[allow(dead_code)] // most fields exist only to give the read ones their offsets
    struct TaskVmInfo {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        // rev 1
        phys_footprint: u64,
        // rev 2
        min_address: u64,
        max_address: u64,
        // rev 3: ledger_phys_footprint_peak is the first of 21 ledger values.
        ledgers: [i64; 21],
        // rev 4
        limit_bytes_remaining: u64,
    }

    const TASK_VM_INFO: u32 = 22;
    const KERN_SUCCESS: i32 = 0;

    // Stable libSystem ABI, declared here rather than through `libc`, whose
    // Mach bindings are deprecated in favour of a separate crate.
    // `mach_task_self()` in C is a macro that reads `mach_task_self_`.
    extern "C" {
        static mach_task_self_: u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut i32,
            count: *mut u32,
        ) -> i32;
    }

    /// Number of `natural_t` (4-byte) words up to and including a field.
    const fn words_through(offset: usize, size: usize) -> u32 {
        ((offset + size) / 4) as u32
    }

    /// Fills the struct and returns it with the word count the kernel wrote.
    fn task_vm_info() -> Option<(TaskVmInfo, u32)> {
        // SAFETY: TaskVmInfo is plain integers, so all-zero is a valid value.
        let mut info: TaskVmInfo = unsafe { std::mem::zeroed() };
        let mut count = (size_of::<TaskVmInfo>() / 4) as u32;
        // SAFETY: `mach_task_self_` is initialised by libSystem before main.
        // `info` is writable for `count` 4-byte words and outlives the call;
        // task_info writes at most `count` words and updates it.
        let kr = unsafe {
            task_info(
                mach_task_self_,
                TASK_VM_INFO,
                (&mut info as *mut TaskVmInfo).cast(),
                &mut count,
            )
        };
        (kr == KERN_SUCCESS).then_some((info, count))
    }

    pub fn footprint_bytes() -> Option<u64> {
        let (info, count) = task_vm_info()?;
        let needed = words_through(offset_of!(TaskVmInfo, phys_footprint), 8);
        (count >= needed).then_some(info.phys_footprint)
    }

    pub fn peak_footprint_bytes() -> Option<u64> {
        let (info, count) = task_vm_info()?;
        let needed = words_through(offset_of!(TaskVmInfo, ledgers), 8);
        if count < needed {
            return None;
        }
        let ledgers = info.ledgers;
        u64::try_from(ledgers[0]).ok()
    }

    #[cfg(target_os = "ios")]
    pub fn available_bytes() -> Option<u64> {
        extern "C" {
            /// `<os/proc.h>`, iOS 13+: bytes the app can still allocate before
            /// hitting its memory limit. Returns 0 when there is no limit.
            fn os_proc_available_memory() -> usize;
        }
        // SAFETY: no arguments, no preconditions; reads kernel state only.
        let bytes = unsafe { os_proc_available_memory() } as u64;
        (bytes > 0).then_some(bytes)
    }

    /// macOS has no per-process limit: report free + inactive pages, the
    /// figure the GGUF loader has always used to decide on mmap.
    #[cfg(target_os = "macos")]
    pub fn available_bytes() -> Option<u64> {
        let sysctl = |name: &str| -> Option<u64> {
            let out = std::process::Command::new("sysctl")
                .args(["-n", name])
                .output()
                .ok()?;
            String::from_utf8(out.stdout).ok()?.trim().parse().ok()
        };
        let page_size = sysctl("hw.pagesize").unwrap_or(16384);
        let free = sysctl("vm.page_free_count").unwrap_or(0);
        let inactive = sysctl("vm.page_inactive_count").unwrap_or(0);
        (free > 0 || inactive > 0).then_some((free + inactive) * page_size)
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod imp {
    /// A `kB` value from a `/proc` key/value file, in bytes.
    fn proc_kb(path: &str, key: &str) -> Option<u64> {
        let text = std::fs::read_to_string(path).ok()?;
        let rest = text.lines().find_map(|l| l.strip_prefix(key))?;
        let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kb * 1024)
    }

    pub fn footprint_bytes() -> Option<u64> {
        proc_kb("/proc/self/status", "VmRSS:")
    }

    pub fn peak_footprint_bytes() -> Option<u64> {
        proc_kb("/proc/self/status", "VmHWM:")
    }

    pub fn available_bytes() -> Option<u64> {
        proc_kb("/proc/meminfo", "MemAvailable:")
    }
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "android"
)))]
mod imp {
    pub fn footprint_bytes() -> Option<u64> {
        None
    }

    pub fn peak_footprint_bytes() -> Option<u64> {
        None
    }

    pub fn available_bytes() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn footprint_tracks_a_touched_allocation() {
        const SIZE: usize = 64 << 20;
        let before = footprint_bytes().expect("footprint");
        // Touch every page so the allocation is really resident.
        let block = vec![1u8; SIZE];
        let during = footprint_bytes().expect("footprint");
        let peak = peak_footprint_bytes().expect("peak footprint");
        assert!(block.iter().step_by(4096).all(|&b| b == 1));
        assert!(
            during >= before + (SIZE as u64) * 3 / 4,
            "footprint grew {} MB for a 64 MB allocation",
            (during.saturating_sub(before)) >> 20
        );
        assert!(peak >= during, "peak {peak} < current {during}");
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn available_memory_is_reported() {
        assert!(available_bytes().is_some_and(|b| b > 0));
    }
}
