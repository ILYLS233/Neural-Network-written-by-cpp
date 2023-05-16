#ifndef NEURALNETWORKGUI_H
#define NEURALNETWORKGUI_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QMessageBox>
#include<qtextedit.h>
#include<qfiledialog.h>

namespace Ui {
class NeuralNetworkGUI;
}

class NeuralNetworkGUI : public QMainWindow
{
    Q_OBJECT

public:
    explicit NeuralNetworkGUI(QWidget *parent = 0);
    ~NeuralNetworkGUI();

private:
    void train();
    void test();
    void openFolder();
    void openFile();
    void clearOut();
    Ui::NeuralNetworkGUI *ui;
    QLabel *filename_label;   // 文件名标签
    QLineEdit *filename_edit; // 文件名输入框
    QPushButton *train_button; // 训练按钮
    QPushButton *test_button; // 测试按钮
    QPushButton *open_file_button; //输入文件按钮
    QPushButton *clear_button; //清空按钮
    QTextEdit *output_text;   // 输出文本框
    QString filename;

};

#endif // NEURALNETWORKGUI_H
