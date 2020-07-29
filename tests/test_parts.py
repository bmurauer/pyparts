from pypart import Part, pick_one, sequential, union
from tests.utils import PrintingTransformer


def test_sequential():
    _, params = sequential('sequential', [
        Part('t0', PrintingTransformer('t0'), {'name': ['t0', 'T0']}),
        Part('t1', PrintingTransformer('t1')),
        Part('t2', PrintingTransformer('t2')),
    ])

    expected_params = {
        't0__name': ['t0', 'T0'],
    }

    assert params == expected_params


def test_pick_one():
    _, params = pick_one('pick_one', [
        Part('t0', PrintingTransformer('t0'), {'name': ['t0', 'T0']}),
        Part('t1', PrintingTransformer('t1')),
        Part('t2', PrintingTransformer('t2')),
    ])

    expected_params = {
        'selected_model': [
            ('t0', {'name': 't0'}),
            ('t0', {'name': 'T0'}),
            ('t1', {}),
            ('t2', {}),
        ],
    }

    assert params == expected_params


def test_optional():
    _, params = pick_one('pick_one', [
        Part('t0', PrintingTransformer('t0'), {'name': ['t0', 'T0']}),
        Part('t1', PrintingTransformer('t1')),
    ], optional=True)

    expected_params = {
        'selected_model': [
            ('t0', {'name': 't0'}),
            ('t0', {'name': 'T0'}),
            ('t1', {}),
            (None, {}),
        ],
    }

    assert params == expected_params


def test_nested():
    _, params = pick_one('outer', [
        sequential('s1', [
            Part('t0', PrintingTransformer('T0')),
            pick_one('inner', [
                Part(
                    't1',
                    PrintingTransformer('T1'),
                    {'name': ['T1_OINS', 'T1_ZWOI']},
                ),
                Part('t2', PrintingTransformer('T2')),
                Part('None', None),
            ]),
            Part('t3', PrintingTransformer('T3')),
        ])
    ])

    expected_params = {
        'selected_model': [
            ('s1', {'inner__selected_model': ('t1', {'name': 'T1_OINS'})}),
            ('s1', {'inner__selected_model': ('t1', {'name': 'T1_ZWOI'})}),
            ('s1', {'inner__selected_model': ('t2', {})}),
            ('s1', {'inner__selected_model': ('None', {})}),
        ]
    }

    assert params == expected_params


def test_union():
    model, params = sequential('root', [
        union('features', [
            Part('t0', PrintingTransformer('T0')),
            Part('t2', PrintingTransformer('T2'), {
                'name': ['T3', 'T4'],
            }),
        ]).add_params({
            'transformer_weights': [
                {'t0': 0, 't1': 1},
                {'t0': 1, 't1': 1},
                {'t0': 1, 't1': 0},
            ]
        })
    ])

    expected_params = {
        'features__transformer_weights': [
            {'t0': 0, 't1': 1},
            {'t0': 1, 't1': 1},
            {'t0': 1, 't1': 0},
        ],
        'features__t2__name': ['T3', 'T4']
    }

    assert params == expected_params
