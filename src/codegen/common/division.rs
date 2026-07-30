#[derive(Debug, Clone, Copy)]
pub(crate) struct SignedMagic {
    pub(crate) multiplier: i32,
    pub(crate) shift: u32,
    pub(crate) add_dividend: bool,
}

/// Hacker's Delight, chapter 10: magic multiplier for signed division by a
/// positive, non-power-of-two 32-bit constant.
pub(crate) fn signed_magic_positive(divisor: u32) -> SignedMagic {
    debug_assert!((2..=i32::MAX as u32).contains(&divisor));
    debug_assert!(!divisor.is_power_of_two());

    let divisor = u64::from(divisor);
    let two31 = 1u64 << 31;
    let anc = two31 - 1 - two31 % divisor;
    let mut p = 31u32;
    let (mut q1, mut r1) = (two31 / anc, two31 % anc);
    let (mut q2, mut r2) = (two31 / divisor, two31 % divisor);

    loop {
        p += 1;
        q1 <<= 1;
        r1 <<= 1;
        if r1 >= anc {
            q1 += 1;
            r1 -= anc;
        }
        q2 <<= 1;
        r2 <<= 1;
        if r2 >= divisor {
            q2 += 1;
            r2 -= divisor;
        }

        let delta = divisor - r2;
        if q1 > delta || (q1 == delta && r1 != 0) {
            break;
        }
    }

    let multiplier = (q2 + 1) as u32 as i32;
    SignedMagic {
        multiplier,
        shift: p - 32,
        add_dividend: multiplier < 0,
    }
}
