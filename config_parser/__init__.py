from .parser import create_config

# config_parser 폴더 내에 parser.py 에서 create_config 모듈을 가져옴
# = config_parser 폴더 내에 create_config가 있는 코드가 있어야 함

## __init__.py은 따로 호출할 필요 없이 / 패키지에 접근하면 자동으로 실행됨
## 즉. config_parser가 import되었기 때문에 자동적으로 __init__.py가 실행됨
