import dsl


DSL_DICT = {
    ('list', 'list') : [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
    ('list', 'atom') : [dsl.FoldFunction, dsl.SimpleITE],
    ('atom', 'atom') : [dsl.NOSGradSelectF,
                        dsl.NOS1stMomSelectF,
                        dsl.NOSNoiseSelectF,
                        
                        dsl.AddFunction,
                        dsl.SubFunction,
                        dsl.MultiplyFunction,
                        dsl.DivideFunction,
                        dsl.ExpBinaryFunction,
                        
                        dsl.SignFunction,
                        dsl.SqrtABSFunction,
                        dsl.ExpFunction
                        ]
}

CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}
