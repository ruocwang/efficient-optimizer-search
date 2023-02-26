


def update_config(config, extra_configs):
    if extra_configs == "none" or extra_configs == "":
        return config
    extra_configs = extra_configs.split(';')
    for extra_config in extra_configs:
        cfg_name, cfg_new_val = extra_config.split('=')
        if cfg_name not in config:
            print(f'\n[WARNING]: {cfg_name} was not in the original config !!!!!!!!!!!!!!\n')
        try:
            res = eval(cfg_new_val)
            if isinstance(res, dict):
                config[cfg_name].update(res)
            else:
                config[cfg_name] = res
        except:
            config[cfg_name] = cfg_new_val
    
    return config