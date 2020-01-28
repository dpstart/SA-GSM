from convnets import resnet18

class VideoModel():
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet18', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, crop_num=1, partial_bn=True, print_spec=True, gsm=False, target_transform=None):
        super(VideoModel, self).__init__()

        self.modality = modality
        self.num_segments = num_segments
        print('Number of segments = {}'.format(self.num_segments))
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.gsm = gsm
        self.target_transform = target_transform


        self._prepare_base_model(base_model)


    def _prepare_base_model(self, base_model):
        import model_zoo
        if base_model == 'resnet18':
            if self.gsm:
                pass
            else:
                base_model = resnet18
