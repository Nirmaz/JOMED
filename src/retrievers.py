
import copy
import torch
import open_clip

EMBEDDINGS_DIM: int = 768

class VisionRetrieverOpenClip(torch.nn.Module):
    def __init__(self, model):
        super(VisionRetrieverOpenClip, self).__init__()
        self.model = model

    def forward(
        self,
            image = None,
    ):
        image_features = self.model.encode_image(image)
        image_features = image_features / torch.linalg.norm(image_features,dim=-1, keepdim=True)

        return image_features

class TextRetrieverOpenClip(torch.nn.Module):
    def __init__(self, model):
        super(TextRetrieverOpenClip, self).__init__()
        self.model = model

    def forward(
        self,
           text = None,
    ):
        text_features = self.model.encode_text(text.squeeze(dim=1))
        text_features = text_features / torch.linalg.norm(text_features,dim=-1, keepdim=True)

        return text_features

class VisionRetrieverJina(torch.nn.Module):
    def __init__(self, model):
        super(VisionRetrieverJina, self).__init__()
        self.model = model

    def forward(
        self,
            image = None,
    ):
        image_features = self.model.vision_model(image)
        image_features = image_features / torch.linalg.norm(image_features,dim=-1, keepdim=True)

        return image_features

class TextRetrieverJina(torch.nn.Module):
    def __init__(self, model):
        super(TextRetrieverJina, self).__init__()
        self.model = model

    def forward(
        self,
           text = None,
    ):
        text_features = self.model.text_model(text.squeeze(dim=1))
        text_features = text_features / torch.linalg.norm(text_features,dim=-1, keepdim=True)

        return text_features



class BaseRetriever(torch.nn.Module):
    """A retriever needs to be able to embed queries and passages, and have a forward function"""

    def __init__(self, *args, **kwargs):
        super(BaseRetriever, self).__init__()

    def embed_queries(self, *args, **kwargs):
        raise NotImplementedError()

    def embed_passages(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, is_passages=False, **kwargs):
        if is_passages:
            return self.embed_passages(*args, **kwargs)
        else:
            return self.embed_queries(*args, **kwargs)

    def gradient_checkpointing_enable(self):
        for m in self.children():
            m.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        for m in self.children():
            m.gradient_checkpointing_disable()


class DualEncoderRetriever(BaseRetriever):
    """Wrapper for standard contriever, or other dual encoders that parameter-share"""

    def __init__(self, opt, contriever):
        super(DualEncoderRetriever, self).__init__()
        self.opt = opt
        self.contriever = contriever

    def _embed(self, *args, **kwargs):
        return self.contriever(*args, **kwargs)

    def embed_queries(self, *args, **kwargs):
        return self._embed(*args, **kwargs)

    def embed_passages(self, *args, **kwargs):
        return self._embed(*args, **kwargs)


class UntiedDualEncoderRetriever(BaseRetriever):
    """Like DualEncoderRetriever, but dedicated encoders for passage and query embedding"""

    def __init__(self, opt, query_encoder, passage_encoder=None):
        """Create the module: if passage_encoder is none, one will be created as a deep copy of query_encoder"""
        super(UntiedDualEncoderRetriever, self).__init__()
        self.opt = opt
        self.query_contriever = query_encoder
        if passage_encoder is None:
            passage_encoder = copy.deepcopy(query_encoder) if hasattr(query_encoder, "module") else query_encoder
        self.passage_contriever = passage_encoder

    def embed_queries(self, *args, **kwargs):
        return self.query_contriever(*args, **kwargs)

    def embed_passages(self, *args, **kwargs):
        if self.opt.query_side_retriever_training:
            is_train = self.passage_contriever.training
            self.passage_contriever.eval()
            with torch.no_grad():
                passage_emb = self.passage_contriever(*args, **kwargs)
            if is_train:
                self.passage_contriever.train()

        else:
            passage_emb = self.passage_contriever(*args, **kwargs)

        return passage_emb




