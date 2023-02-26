

def get_search_space(dsl_name):
    if dsl_name == 'mnistnet':
        from .dsl_mnistnet import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv':
        from .dsl_conv import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-subset':
        from .dsl_conv_subset import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-nodecay':
        from .dsl_conv_nodecay import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-subset-nodecay':
        from .dsl_conv_subset_nodecay import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    
    elif dsl_name == 'conv-adv-v1':
        from .dsl_conv_adv_v1 import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-adv-v2':
        from .dsl_conv_adv_v2 import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-adv-v3':
        from .dsl_conv_adv_v3 import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB

    elif dsl_name == 'conv-adv-v1-proj':
        from .dsl_conv_adv_v1_proj import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-adv-v1-larger':
        from .dsl_conv_adv_v1_larger import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-adv-v1-smaller':
        from .dsl_conv_adv_v1_smaller import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-adv-v1-ar':
        from .dsl_conv_adv_v1_ar import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-adv-v1-aa':
        from .dsl_conv_adv_v1_aa import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-adv-v1-noaa':
        from .dsl_conv_adv_v1_noaa import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    
    elif dsl_name == 'conv-adv-v1-mp':
        from .dsl_conv_adv_v1_mp import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-adv-v1-am':
        from .dsl_conv_adv_v1_am import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'conv-adv-v1-amp':
        from .dsl_conv_adv_v1_amp import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB

    elif dsl_name == 'conv-adv-v1-mind':
        from .dsl_conv_adv_v1_mind import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB

    elif dsl_name == 'gnn-v1':
        from .dsl_gnn_v1 import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'gnn-v2':
        from .dsl_gnn_v2 import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB

    elif dsl_name == 'const-simple':
        from .dsl_const_simple import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
    elif dsl_name == 'const':
        from .dsl_const import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB

    elif dsl_name == 'all':
        from .dsl_all import DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB

    return DSL_DICT, CUSTOM_EDGE_COSTS, VOCAB
