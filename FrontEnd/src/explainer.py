import main
import gallery
from PyQt5.QtWidgets import QWidget, QPushButton, QStyle, QSlider, QStyleOptionSlider, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt, QPoint, QRect
import matplotlib.pyplot as plt
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

class Explainer(QWidget):
    def __init__(self, stack, id, x_test, y_test, iclf, clfs, values):
        super(Explainer, self).__init__()
        back = QPushButton("Back", self)
        self.hpSlider = self.LabeledSlider(minimax=(0, len(values)-1), labels=values, parent=self)
        self.go = QPushButton("Go", self)
        self.go.setGeometry(380, 200, 100, 32)
        self.hpSlider.move(30, 60)
        back.clicked.connect(lambda: main.transition(stack, gallery.Gallery(stack, x_test, y_test, iclf, clfs)))
        self.go.clicked.connect(lambda: self.printThenExplain(id, x_test, y_test, iclf, clfs))
        self.show()
    
    def printThenExplain(self, id, x_test, y_test, iclf, clfs):

        # self.explain(id, x_test, y_test, clfs[self.slider1.value()][self.slider2.value()][self.slider3.value()])
        self.explain(id, x_test, y_test, iclf, clfs[self.hpSlider.sl.value()])

    def explain(self, id, x_test, y_test, iclf, vclf):
        explainer = lime_image.LimeImageExplainer(verbose = False)
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        explanation = explainer.explain_instance(x_test[id], 
                                classifier_fn = iclf.predict_proba, 
                                top_labels=10, hide_color=0, num_samples=2000, segmentation_fn=segmenter)
        temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
        ax1.imshow(mark_boundaries(temp, mask))
        # explainer = lime_image.LimeImageExplainer(verbose = False)
        # segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        explanation = explainer.explain_instance(x_test[id], 
                                        classifier_fn = vclf.predict_proba, 
                                        top_labels=10, hide_color=0, num_samples=2000, segmentation_fn=segmenter)
        # temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=True, num_features=5, hide_rest=True, min_weight = 0.01)
        # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
        # ax1.imshow(mark_boundaries(temp, mask))
        temp, mask = explanation.get_image_and_mask(y_test[id], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
        # print(y_test[id])
        # print(vclf.predict(x_test)[id])
        ax2.imshow(mark_boundaries(temp, mask))
        ax1.axis('off')
        ax2.axis('off')
        plt.show()



    # class LabeledSlider(QWidget):
    #     def __init__(self, minimax, interval=1, orientation=Qt.Vertical,
    #             labels=None, parent=None):
    #         super(Explainer.LabeledSlider, self).__init__(parent=parent)

    #         levels=range(minimax[0], minimax[1]+interval, interval)
    #         if labels is not None:
    #             if not isinstance(labels, (tuple, list)):
    #                 raise Exception("<labels> is a list or tuple.")
    #             if len(labels) != len(levels):
    #                 raise Exception("Size of <labels> doesn't match levels.")
    #             self.levels=list(zip(levels,labels))
    #         else:
    #             self.levels=list(zip(levels,map(str,levels)))

    #         if orientation==Qt.Horizontal:
    #             self.layout=QVBoxLayout(self)
    #         elif orientation==Qt.Vertical:
    #             self.layout=QHBoxLayout(self)
    #         else:
    #             raise Exception("<orientation> wrong.")

    #         # gives some space to print labels
    #         self.left_margin=10
    #         self.top_margin=10
    #         self.right_margin=10
    #         self.bottom_margin=10

    #         self.layout.setContentsMargins(self.left_margin,self.top_margin,
    #                 self.right_margin,self.bottom_margin)

    #         self.sl=QSlider(orientation, self)
    #         self.sl.setMinimum(minimax[0])
    #         self.sl.setMaximum(minimax[1])
    #         self.sl.setValue(minimax[0])
    #         if orientation==Qt.Horizontal:
    #             self.sl.setTickPosition(QSlider.TicksBelow)
    #             self.sl.setMinimumWidth(300) # just to make it easier to read
    #         else:
    #             self.sl.setTickPosition(QSlider.TicksLeft)
    #             self.sl.setMinimumHeight(300) # just to make it easier to read
    #         self.sl.setTickInterval(interval)
    #         self.sl.setSingleStep(1)

    #         self.layout.addWidget(self.sl)

    #     def paintEvent(self, e):

    #         super(Explainer.LabeledSlider,self).paintEvent(e)
        
    #         print(self.levels)
    #         style=self.sl.style()
    #         painter=QPainter(self)
    #         st_slider=QStyleOptionSlider()
    #         st_slider.initFrom(self.sl)
    #         st_slider.orientation=self.sl.orientation()

    #         length=style.pixelMetric(QStyle.PM_SliderLength, st_slider, self.sl)
    #         available=style.pixelMetric(QStyle.PM_SliderSpaceAvailable, st_slider, self.sl)

    #         # for v, v_str in self.levels:
    #         #     print(v_str)
    #         #     # get the size of the label
    #         #     # rect=painter.drawText(QRect(), Qt.TextDontPrint, v_str)
    #         #     # rect=painter.drawText(QRect(), Qt.TextDontPrint, text=v_str)
    #         #     # rect=painter.drawText(QRect(), 16384, v_str)

    #         #     if self.sl.orientation()==Qt.Horizontal:
    #         #         # I assume the offset is half the length of slider, therefore
    #         #         # + length//2
    #         #         x_loc=QStyle.sliderPositionFromValue(self.sl.minimum(),
    #         #                 self.sl.maximum(), v, available)+length//2

    #         #         # left bound of the text = center - half of text width + L_margin
    #         #         left=x_loc-rect.width()//2+self.left_margin
    #         #         bottom=self.rect().bottom()

    #         #         # enlarge margins if clipping
    #         #         if v==self.sl.minimum():
    #         #             if left<=0:
    #         #                 self.left_margin=rect.width()//2-x_loc
    #         #             if self.bottom_margin<=rect.height():
    #         #                 self.bottom_margin=rect.height()

    #         #             self.layout.setContentsMargins(self.left_margin,
    #         #                     self.top_margin, self.right_margin,
    #         #                     self.bottom_margin)

    #         #         if v==self.sl.maximum() and rect.width()//2>=self.right_margin:
    #         #             self.right_margin=rect.width()//2
    #         #             self.layout.setContentsMargins(self.left_margin,
    #         #                     self.top_margin, self.right_margin,
    #         #                     self.bottom_margin)

    #         #     else:
    #         #         y_loc=QStyle.sliderPositionFromValue(self.sl.minimum(),
    #         #                 self.sl.maximum(), v, available, upsideDown=True)

    #         #         bottom=y_loc+length//2+rect.height()//2+self.top_margin-3
    #         #         # there is a 3 px offset that I can't attribute to any metric

    #         #         left=self.left_margin-rect.width()
    #         #         if left<=0:
    #         #             self.left_margin=rect.width()+2
    #         #             self.layout.setContentsMargins(self.left_margin,
    #         #                     self.top_margin, self.right_margin,
    #         #                     self.bottom_margin)

    #         #     pos=QPoint(left, bottom)
    #         #     painter.drawText(pos, v_str)

    #         # return


    # if __name__ == '__main__':
    #     app = QtWidgets.QApplication(sys.argv)
    #     frame=QtWidgets.QWidget()
    #     ha=QtWidgets.QHBoxLayout()
    #     frame.setLayout(ha)

    #     w = LabeledSlider(1, 10 , 1, orientation=Qt.Horizontal)

    #     ha.addWidget(w)
    #     frame.show()
    #     sys.exit(app.exec_())




    
    class LabeledSlider(QWidget):
        def __init__(self, minimax, interval=1, orientation=Qt.Vertical,
                labels=None, p0=0, parent=None):
            super(Explainer.LabeledSlider, self).__init__(parent=parent)

            levels=range(minimax[0], minimax[1] + interval, interval)

            if labels is not None:
                if not isinstance(labels, (tuple, list)):
                    raise Exception("<labels> is a list or tuple.")
                if len(labels) != len(levels):
                    raise Exception("Size of <labels> doesn't match levels.")
                self.levels=list(zip(levels,labels))
            else:
                self.levels=list(zip(levels,map(str,levels)))

            if orientation==Qt.Horizontal:
                self.layout=QVBoxLayout(self)
            elif orientation==Qt.Vertical:
                self.layout=QHBoxLayout(self)
            else:
                raise Exception("<orientation> wrong.")

            # gives some space to print labels
            self.left_margin=10
            self.top_margin=10
            self.right_margin=10
            self.bottom_margin=10

            self.layout.setContentsMargins(self.left_margin,self.top_margin,
                    self.right_margin,self.bottom_margin)

            self.sl=QSlider(orientation, self)
            self.sl.setMinimum(minimax[0])
            self.sl.setMaximum(minimax[1])
            self.sl.setValue(minimax[0])
            self.sl.setSliderPosition(p0)
            if orientation==Qt.Horizontal:
                self.sl.setTickPosition(QSlider.TicksBelow)
                self.sl.setMinimumWidth(300) # just to make it easier to read
            else:
                self.sl.setTickPosition(QSlider.TicksLeft)
                self.sl.setMinimumHeight(300) # just to make it easier to read
            self.sl.setTickInterval(interval)
            self.sl.setSingleStep(1)

            self.layout.addWidget(self.sl)

        def paintEvent(self, e):

            super(Explainer.LabeledSlider,self).paintEvent(e)
            style=self.sl.style()
            painter=QPainter(self)
            st_slider=QStyleOptionSlider()
            st_slider.initFrom(self.sl)
            st_slider.orientation=self.sl.orientation()

            length=style.pixelMetric(QStyle.PM_SliderLength, st_slider, self.sl)
            available=style.pixelMetric(QStyle.PM_SliderSpaceAvailable, st_slider, self.sl)

            for v, v_str in self.levels:

                # get the size of the label
                rect=painter.drawText(QRect(), Qt.TextDontPrint, str(v_str))

                if self.sl.orientation()==Qt.Horizontal:
                    # I assume the offset is half the length of slider, therefore
                    # + length//2
                    x_loc=QStyle.sliderPositionFromValue(self.sl.minimum(),
                            self.sl.maximum(), v, available)+length//2

                    # left bound of the text = center - half of text width + L_margin
                    left=x_loc-rect.width()//2+self.left_margin
                    bottom=self.rect().bottom()

                    # enlarge margins if clipping
                    if v==self.sl.minimum():
                        if left<=0:
                            self.left_margin=rect.width()//2-x_loc
                        if self.bottom_margin<=rect.height():
                            self.bottom_margin=rect.height()

                        self.layout.setContentsMargins(self.left_margin,
                                self.top_margin, self.right_margin,
                                self.bottom_margin)

                    if v==self.sl.maximum() and rect.width()//2>=self.right_margin:
                        self.right_margin=rect.width()//2
                        self.layout.setContentsMargins(self.left_margin,
                                self.top_margin, self.right_margin,
                                self.bottom_margin)

                else:
                    y_loc=QStyle.sliderPositionFromValue(self.sl.minimum(),
                            self.sl.maximum(), v, available, upsideDown=True)

                    bottom=y_loc+length//2+rect.height()//2+self.top_margin-3
                    # there is a 3 px offset that I can't attribute to any metric

                    left=self.left_margin-rect.width()
                    if left<=0:
                        self.left_margin=rect.width()+2
                        self.layout.setContentsMargins(self.left_margin,
                                self.top_margin, self.right_margin,
                                self.bottom_margin)

                pos=QPoint(left, bottom)
                painter.drawText(pos, str(v_str))

            return