#include "neuralnetworkgui.h"
#include <QApplication>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    NeuralNetworkGUI w;
    w.show();

    return a.exec();
}
