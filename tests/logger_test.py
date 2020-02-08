from logger import Logger


def test_get_all_models():
    models = Logger.get_all_models('/work/pfeiffer/master/groupnorm/152/')
    print(models)
    assert len(models) == 3


def test_get_cfg_path():
    cfg = Logger.get_cfg_path('/work/pfeiffer/master/groupnorm/152/')
    print(cfg)
