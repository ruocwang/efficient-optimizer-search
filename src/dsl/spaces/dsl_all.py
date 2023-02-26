import dsl.functions as dsl_func


DSL_DICT = {
    ('atom', 'atom') : [
                        #### binary (done)
                        dsl_func.AddFunction,
                        dsl_func.SubFunction,
                        dsl_func.MulFunction,
                        dsl_func.DivFunction,
                        dsl_func.PowFunction, ## not included in nos-rep
                        
                        #### unary (done)
                        dsl_func.NegFunction,
                        dsl_func.ExpFunction,
                        dsl_func.LogFunction,
                        dsl_func.SqrtFunction,
                        
                        dsl_func.Clip5Function,
                        dsl_func.Clip4Function,
                        dsl_func.Clip3Function,
                        dsl_func.Clip0Function,
                        
                        dsl_func.Drop1Function,
                        dsl_func.Drop3Function,
                        dsl_func.Drop5Function,
                        
                        dsl_func.SignFunction,
                    
                        #### operands
                        dsl_func.CIFAR_Grad,
                        dsl_func.CIFAR_Grad2,
                        dsl_func.CIFAR_Grad3,
                        # dsl_func.CIFAR_GradPrev,

                        dsl_func.CIFAR_Mom1,
                        dsl_func.CIFAR_Mom2,
                        dsl_func.CIFAR_Mom3,

                        dsl_func.CIFAR_Mom1p,
                        dsl_func.CIFAR_Mom2p,
                        dsl_func.CIFAR_Mom3p,

                        dsl_func.CIFAR_AdaMom1,
                        dsl_func.CIFAR_AdaMom2,
                        dsl_func.CIFAR_AdaMom3,

                        dsl_func.CIFAR_AdaMom1p,
                        dsl_func.CIFAR_AdaMom2p,
                        dsl_func.CIFAR_AdaMom3p,

                        dsl_func.CIFAR_SignGrad,
                        dsl_func.CIFAR_SignMom1,
                        dsl_func.CIFAR_SignMom1p,
                        
                        dsl_func.N1,
                        dsl_func.N2,
                        dsl_func.Noise,
                        
                        dsl_func.W1,
                        dsl_func.W2,
                        dsl_func.W3,
                        dsl_func.W4,
                        
                        dsl_func.Adam,
                        dsl_func.RMSprop,
                        
                        dsl_func.LinearDecay,
                        dsl_func.CosineDecay,
                        dsl_func.RestartDecay,
                        dsl_func.AnnealedNoiseDecay,
                        dsl_func.DynamicDecay,

                        dsl_func.LinearDecayMin,
                        dsl_func.CosineDecayMin,
                        dsl_func.RestartDecayMin,
                        dsl_func.AnnealedNoiseDecayMin,
                        ],
}


CUSTOM_EDGE_COSTS = {
    ('list', 'list')   : {},
    ('list', 'atom')   : {},
    ('atom', 'atom')   : {},
    # ('const', 'const') : {}
}

VOCAB = {}
idx = 1 ## 0 is reserved for padding/empty
for io_type in DSL_DICT:
    for func in DSL_DICT[io_type]:
        func_name = func.get_name()
        VOCAB[func_name] = idx; idx += 1