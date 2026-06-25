pub(super) fn mov_w_imm(reg: &str, value: i32) -> String {
    let mut out = String::new();
    let value = value as u32;
    let mut emitted = false;
    for idx in 0..2 {
        let part = ((value >> (idx * 16)) & 0xffff) as u16;
        if part == 0 && emitted {
            continue;
        }
        if !emitted {
            out.push_str(&format!("  movz {}, #{}, lsl #{}\n", reg, part, idx * 16));
            emitted = true;
        } else if part != 0 {
            out.push_str(&format!("  movk {}, #{}, lsl #{}\n", reg, part, idx * 16));
        }
    }
    if !emitted {
        out.push_str(&format!("  movz {}, #0\n", reg));
    }
    out
}

pub(super) fn mov_x_imm(reg: &str, value: i64) -> String {
    let mut out = String::new();
    let value = value as u64;
    let mut emitted = false;
    for idx in 0..4 {
        let part = ((value >> (idx * 16)) & 0xffff) as u16;
        if part == 0 && emitted {
            continue;
        }
        if !emitted {
            out.push_str(&format!("  movz {}, #{}, lsl #{}\n", reg, part, idx * 16));
            emitted = true;
        } else if part != 0 {
            out.push_str(&format!("  movk {}, #{}, lsl #{}\n", reg, part, idx * 16));
        }
    }
    if !emitted {
        out.push_str(&format!("  movz {}, #0\n", reg));
    }
    out
}
