import logging

from .Basic import BasicFormatter
from .squadPromptT5Formatter import squadPromptT5Formatter
from .squadclosedPromptT5Formatter import squadclosedPromptT5Formatter
from .nq_openPromptT5Formatter import nq_openPromptT5Formatter
from .tqaclosedPromptT5Formatter import tqaclosedPromptT5Formatter
from .tqaPromptT5Formatter import tqaPromptT5Formatter
from .wqPromptT5Formatter import wqPromptT5Formatter
from .duorcPromptT5Formatter import duorcPromptT5Formatter
from .news_qaPromptT5Formatter import news_qaPromptT5Formatter
from .search_qaPromptT5Formatter import search_qaPromptT5Formatter
from .boolqPromptT5Formatter import boolqPromptT5Formatter
from .boolqclosedPromptT5Formatter import boolqclosedPromptT5Formatter
from .multircPromptT5Formatter import multircPromptT5Formatter
from .multircclosedPromptT5Formatter import multircclosedPromptT5Formatter
from .cosmos_qaPromptT5Formatter import cosmos_qaPromptT5Formatter
from .cosmos_qaclosedPromptT5Formatter import cosmos_qaclosedPromptT5Formatter
from .social_i_qaPromptT5Formatter import social_i_qaPromptT5Formatter
from .social_i_qaclosedPromptT5Formatter import social_i_qaclosedPromptT5Formatter


logger = logging.getLogger(__name__)


formatter_list = {
    "Basic": BasicFormatter,
    "squadPromptT5": squadPromptT5Formatter,
    "squadclosedPromptT5": squadclosedPromptT5Formatter,
    "nq_openPromptT5": nq_openPromptT5Formatter,
    "tqaclosedPromptT5": tqaclosedPromptT5Formatter,
    "tqaPromptT5": tqaPromptT5Formatter,
    "wqPromptT5": wqPromptT5Formatter,
    "duorcPromptT5": duorcPromptT5Formatter,
    "news_qaPromptT5": news_qaPromptT5Formatter,
    "search_qaPromptT5": search_qaPromptT5Formatter,
    "boolqPromptT5": boolqPromptT5Formatter,
    "boolqclosedPromptT5": boolqclosedPromptT5Formatter,
    "multircPromptT5": multircPromptT5Formatter,
    "multircclosedPromptT5": multircclosedPromptT5Formatter,
    "cosmos_qaPromptT5": cosmos_qaPromptT5Formatter,
    "cosmos_qaclosedPromptT5": cosmos_qaclosedPromptT5Formatter,
    "social_i_qaPromptT5": social_i_qaPromptT5Formatter,
    "social_i_qaclosedPromptT5": social_i_qaclosedPromptT5Formatter,
}


def init_formatter(config, mode, *args, **params):
    print("==========================")
    print("init formatter")
    print(mode)

    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning("[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    print("which : ", which)
    print("==========================")

    if which in formatter_list:
        print(which)
        formatter = formatter_list[which](config, mode, *args, **params)
        return formatter
    else:
        logger.error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError
