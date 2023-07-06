import main
import gallery
from PyQt5.QtWidgets import QWidget, QPushButton
from PyQt5.uic import loadUi
import matplotlib.pyplot as plt
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

class Explainer(QWidget):
    def __init__(self, stack, id, x_test, y_test, clfs):
        #Sets labels etc
        super(Explainer, self).__init__()
        loadUi('FrontEnd/UI/Explainer.ui', self)
        back = QPushButton("Back", self)
        back.clicked.connect(lambda: main.transition(stack, gallery.Gallery(stack, x_test, y_test, clfs)))
        self.go.clicked.connect(lambda: self.printThenExplain(id, x_test, y_test, clfs))
        self.show()
    
    def printThenExplain(self, id, x_test, y_test, clfs):
        self.explain(id, x_test, y_test, clfs[self.slider1.value()][self.slider2.value()][self.slider3.value()])

    def explain(self, id, x_test, y_test, clf):
        explainer = lime_image.LimeImageExplainer(verbose = False)
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        explanation = explainer.explain_instance(x_test[id], 
                                        classifier_fn = clf.predict_proba, 
                                        top_labels=10, hide_color=0, num_samples=2000, segmentation_fn=segmenter)

        temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=True, num_features=5, hide_rest=True, min_weight = 0.01)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
        # ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
        ax1.imshow(mark_boundaries(temp, mask))
        # ax1.set_title('Positive Regions for {}'.format(y_test[id]))
        temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
        ax2.imshow(mark_boundaries(temp, mask))
        # ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
        # ax2.set_title('Positive/Negative Regions for {}'.format(y_test[id]))
        ax1.axis('off')
        ax2.axis('off')
        plt.show()