from utils import ExitHandler
import time
import pytest
import os
import signal
def test():
    def bye_world():
        print("bye")

    def send_greet(name, ps="Test"):
        print("Hi {}! PS: {}".format(name, ps))

    handler = ExitHandler()
    handler.register(bye_world)
    handler.register(send_greet, "kilian")
    handler.register(send_greet, "kilian", ps="Passed")

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        for _ in range(10):
            time.sleep(1)
            print("running")
        os.kill(os.getpid(), signal.SIGTERM)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0



