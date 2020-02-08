collect_ignore = ["evaluate_test.py"]
def pytest_addoption(parser):
    parser.addoption('--cfg', default=None)

def pytest_generate_tests(metafunc):
    if 'cfg_file' in metafunc.fixturenames:
        if metafunc.config.getoption('cfg'):
            cfg_file = [metafunc.config.getoption('cfg')]
        else:
            #TODO get all cfg_files
            cfg_file = ['./configs/attributes.json']
        metafunc.parametrize('cfg_file', cfg_file)

