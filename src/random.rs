use rand::Rng;

pub fn choose_from_iter<R, I>(rng: &mut R, iter: I) -> Option<I::Item>
    where R: Rng,
          I: Iterator
{
    let mut count: u32 = 0;
    let mut result = None;
    for item in iter {
        count += 1;
        if rng.gen_weighted_bool(count) {
            result = Some(item);
        }
    }
    return result;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::XorShiftRng;

    #[test]
    fn can_choose_from_empty_range() {
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(None, choose_from_iter(&mut rng, 0..0));
    }

    #[test]
    fn can_choose_from_1_item_rang() {
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(Some(0), choose_from_iter(&mut rng, 0..1));
    }

    #[test]
    fn can_choose_from_2_item_rang() {
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(Some(0), choose_from_iter(&mut rng, 0..2));
    }

    #[test]
    fn can_choose_from_3_item_rang() {
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(Some(2), choose_from_iter(&mut rng, 0..3));
    }
}
