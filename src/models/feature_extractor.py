from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod 
    def extract_features(self, image_path):
        pass