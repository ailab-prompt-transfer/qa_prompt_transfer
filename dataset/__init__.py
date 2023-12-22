from dataset.JsonFromFiles import JsonFromFilesDataset
from .squadDataset import squadDataset
from .squadclosedDataset import squadclosedDataset
from .nq_openDataset import nq_openDataset
from .tqaDataset import tqaDataset
from .tqaclosedDataset import tqaclosedDataset
from .wqDataset import wqDataset
from .duorcDataset import duorcDataset
from .news_qaDataset import news_qaDataset
from .search_qaDataset import search_qaDataset
from .boolqDataset import boolqDataset
from .boolqclosedDataset import boolqclosedDataset
from .multircDataset import multircDataset
from .multircclosedDataset import multircclosedDataset
from .cosmos_qaDataset import cosmos_qaDataset
from .cosmos_qaclosedDataset import cosmos_qaclosedDataset
from .social_i_qaDataset import social_i_qaDataset
from .social_i_qaclosedDataset import social_i_qaclosedDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "squad": squadDataset,
    "squadclosed": squadclosedDataset,
    "nq_open": nq_openDataset,
    "tqa": tqaDataset,
    "tqaclosed": tqaclosedDataset,
    "wq": wqDataset,
    "duorc": duorcDataset,
    "news_qa": news_qaDataset,
    "search_qa": search_qaDataset,
    "boolq": boolqDataset,
    "boolqclosed": boolqclosedDataset,
    "multirc": multircDataset,
    "multircclosed": multircclosedDataset,
    "cosmos_qa": cosmos_qaDataset,
    "cosmos_qaclosed": cosmos_qaclosedDataset,
    "social_i_qa": social_i_qaDataset,
    "social_i_qaclosed": social_i_qaclosedDataset,
}
