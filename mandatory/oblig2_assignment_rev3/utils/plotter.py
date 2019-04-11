import matplotlib.pyplot as plt
from time import sleep

class Plotter():
    def __init__(self, modelParam, config):
        self.modelParam = modelParam
        self.config     = config
        self.fig, self.ax = plt.subplots()
        self.fig.show()
        plt.ylabel('Loss')
        plt.xlabel('epoch [#]')
        train_line = plt.plot([],[],color='blue', label='Train', marker='.', linestyle="")
        val_line  = plt.plot([], [], color='red', label='Validation', marker='.', linestyle="")
        plt.legend(handles=[train_line[0], val_line[0]])
        self.ax.set_axisbelow(True)
        self.ax.grid()
        return


    def update(self, current_epoch, loss, mode):
        if mode=='train':
            color = 'b'
        else:
            color = 'r'

        if self.modelParam['inNotebook']:
            self.ax.scatter(current_epoch, loss, c=color)
            self.ax.set_ylim(bottom=0, top=plt.ylim()[1])
            self.fig.canvas.draw()
            sleep(0.1)
        else:
            plt.scatter(current_epoch, loss, c=color)
            plt.ylim(0.0, plt.ylim()[1])
            plt.draw()
            plt.pause(0.01)
        self.save()
        return


    def save(self):
        # path = self._getPath()
        path = 'loss_images/'+self.modelParam['modelName'][:-1]
        plt.savefig(path+'.png')
        return

    def _getPath(self):
        keys = self.config.keys()
        path = 'loss_images/'
        first=1
        for key in keys:
            if first!=1:
                path += '_'
            else:
                first=0
            element = self.config[key]
            if isinstance(element, str):
                path += element
            elif isinstance(element, int):
                path += key+str(element)
            elif isinstance(element, float):
                path += key+str(element)
            elif isinstance(element, list):
                path += ''
                for elm in element:
                    path += str(elm)
            elif isinstance(element, dict):
                path += ''
                for elKey, elVal in element.items():
                    path += str(elKey) + str(elVal).replace('.', '_')
            else:
                raise Exception('Unknown element in config')
        return path
