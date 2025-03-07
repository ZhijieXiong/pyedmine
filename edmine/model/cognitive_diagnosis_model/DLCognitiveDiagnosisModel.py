from abc import abstractmethod

from edmine.model.CognitiveDiagnosisModel import CognitiveDiagnosisModel


class DLCognitiveDiagnosisModel(CognitiveDiagnosisModel):
    @abstractmethod
    def get_predict_loss(self, batch):
        pass

    @abstractmethod
    def get_predict_score(self, batch):
        pass