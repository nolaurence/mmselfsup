import mmcv
import numpy as np
import pandas as pd

from ..builder import DATASOURCES
from .base import BaseDataSource


@DATASOURCES.register_module()
class Severstal(BaseDataSource):

    def load_annotations(self):

        assert isinstance(self.ann_file, str)
        print("loading {self.anno_file} ...")
        ann_file_np = pd.read_csv(self.ann_file).values
        print("Done")
        
        data_infos = []
        
        print("processing severstal ...")
        
        for i, (filename, label) in enumerate(ann_file_np):
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(label, dtype=np.int64)
            info['idx'] = int(i)
            data_infos.append(info)
        
        print("Done")
        # writing your code here.
        return data_infos