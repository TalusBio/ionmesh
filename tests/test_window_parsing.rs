use ionmesh::ms::tdf::{
    FrameInfoBuilder,
    GroupingLevel,
};

#[test]
fn test_dia_pasef() {
    let finfo_b = FrameInfoBuilder::from_dotd_path("tests/data/diapasef_tdf/data.d".into());
    let finfo = finfo_b.build();

    assert!(finfo.is_ok());

    let finfo = finfo.unwrap();

    // The number of ids in `DiaFrameMsMsWindowGroups` + 1 bc 0 is not used
    assert_eq!(finfo.groups.len(), 9);

    assert!(finfo.groups[0].is_none());
    for group in finfo.groups.iter().skip(1) {
        assert!(group.is_some());
    }

    // Make sure the grouping is correctly assigned... for diaPASEF it should
    // be `QuadWindowGroup`
    match finfo.grouping_level {
        GroupingLevel::QuadWindowGroup => {},
        GroupingLevel::WindowGroup => {
            assert!(false);
        },
    }

    // Make sure the grouping is correct.
    // For this diapasef file  is 8 * 2 (8 window groups, 2 isolation windows per group)
    assert_eq!(finfo.row_to_group.iter().max().unwrap(), &(8 * 2));

    // println!("{:?}", finfo);
    // assert!(false)
}

#[test]
fn test_synchro_dia_pasef() {
    let finfo_b = FrameInfoBuilder::from_dotd_path("tests/data/synchropasef_tdf/data.d".into());
    let finfo = finfo_b.build();

    assert!(finfo.is_ok());

    let finfo = finfo.unwrap();

    // The number of ids in `DiaFrameMsMsWindowGroups` + 1, bc 0 is not used
    assert_eq!(finfo.groups.len(), 5);
    assert!(finfo.groups[0].is_none());
    for group in finfo.groups.iter().skip(1) {
        assert!(group.is_some());
    }

    // Make sure the grouping is correctly assigned... for diaPASEF it should
    // be `QuadWindowGroup`
    match finfo.grouping_level {
        GroupingLevel::QuadWindowGroup => {
            assert!(false);
        },
        GroupingLevel::WindowGroup => {},
    }

    // Make sure the grouping is correct.
    assert_eq!(finfo.row_to_group.iter().max().unwrap(), &4);

    // println!("{:?}", finfo);
}
