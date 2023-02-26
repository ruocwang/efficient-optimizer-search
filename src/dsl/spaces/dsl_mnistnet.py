import dsl.functions as dsl_func



DSL_DICT = {
    ('atom', 'atom') : [dsl_func.AddFunction,
                        dsl_func.MulFunction,
                        dsl_func.DivFunction,
                        dsl_func.SignFunction,
                        dsl_func.SqrtFunction,
                        dsl_func.ScaleFunction,
                        dsl_func.MNIST_Grad,
                        dsl_func.MNIST_GradPrev,
                        dsl_func.MNIST_Mom1,
                        dsl_func.MNIST_Mom2,],
    
    ('const', 'const'): [dsl_func.Const0d9,
                         dsl_func.Const0d99,
                         dsl_func.Const0d999,
                         dsl_func.StepDecayFunction]
}

CUSTOM_EDGE_COSTS = {
    ('list', 'list')   : {},
    ('list', 'atom')   : {},
    ('atom', 'atom')   : {},
    ('const', 'const') : {}
}

VOCAB = {}
idx = 0
for io_type in DSL_DICT:
    for func in DSL_DICT[io_type]:
        func_name = func.get_name()
        VOCAB[func_name] = idx; idx += 1