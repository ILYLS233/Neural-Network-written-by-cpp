#include "neuralnetworkgui.h"
#include "ui_neuralnetworkgui.h"
#include "all_class.h"



void showVectorVals(std::string label, std::vector<double> &v, QTextEdit *output_text){
    QString output = QString::fromStdString(label) + " ";
    unsigned size = v.size();
    for(unsigned i = 0; i < size; i++)
    output += QString::number(v[i]) + " ";
    output += "\n";
    output_text->append(output);
}

NeuralNetworkGUI::NeuralNetworkGUI(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::NeuralNetworkGUI)
{
    ui->setupUi(this);
    // 创建界面元素
    filename_label = new QLabel("文件名：", this);
    filename_edit = new QLineEdit(this);
    train_button = new QPushButton("训练并测试", this);
    //test_button = new QPushButton("测试", this);
    output_text = new QTextEdit(this);
    open_file_button = new QPushButton("打开训练文件", this);
    clear_button = new QPushButton("清空输出",this);



    // 设置元素位置和大小
    filename_label->setGeometry(20, 20, 100, 40);
    filename_edit->setGeometry(140, 20, 320, 40);
    train_button->setGeometry(20, 80, 200, 40);
    //test_button->setGeometry(240, 80, 200, 40);
    output_text->setGeometry(20, 140, 920, 360);
    open_file_button->setGeometry(480, 20, 100, 40);
    clear_button->setGeometry(240,80,200,40);




    // 连接按钮的信号和槽函数
    connect(train_button, &QPushButton::clicked, this, &NeuralNetworkGUI::train);
    //connect(test_button, &QPushButton::clicked, this, &NeuralNetworkGUI::test);
    connect(open_file_button, &QPushButton::clicked, this, &NeuralNetworkGUI::openFile);
    connect(clear_button, &QPushButton::clicked, this, &NeuralNetworkGUI::clearOut);

    // 设置窗口大小和标题
    resize(960, 540);
    setWindowTitle("神经网络");

    // 显示窗口
    show();
}
void NeuralNetworkGUI:: train() {
    // 这里是训练函数的代码，将训练输出信息写入到 QTextEdit 对象中
    QString filename = filename_edit->text();

    // 在文本框中追加训练信息
    output_text->append("训练文件：");
    output_text->append(filename);
    output_text->append("开始训练...\n");

    // ...

    std::string s_filename = filename.toStdString();
    TrainingData trainData(s_filename);
    std::vector<unsigned> topology;

    trainData.getTopology(topology);
    Net myNet(topology);

    std::vector<double> inputVals, targetVals, resultVals;

    int trainingPass = 0;
    while(!trainData.isEof()){
        ++trainingPass;
        QString pass = "Pass " + QString::number(trainingPass) + "\n";
        output_text->append(pass);
        if(trainData.getNextInputs(inputVals) != topology[0])
            break;
        showVectorVals(": Inputs :", inputVals,output_text);
        myNet.feedForward(inputVals);
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals,output_text);

        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals,output_text);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        QString loss = "Net recent average loss: " + QString::number(myNet.getRecentAverageloss()) + "\n";
        output_text->append(loss);
    }

    output_text->append("Done\n");
    output_text->append("训练完成后将进行测试，请耐心等待\n");

    // 训练结束后弹出消息框
    QMessageBox::information(this, "训练成功", "神经网络训练成功！");

    // 这里是测试函数的代码，获取测试文件名字并计算准确率
    filename = filename_edit->text();
    QFileInfo trainingInfo(filename);
    filename = trainingInfo.path() + "/testData.txt";

    // 在文本框中追加测试信息
    output_text->append("****************\n测试文件：");
    output_text->append(filename);
    output_text->append("开始测试...\n");

    // ...
    s_filename = filename.toStdString();
    TestData testData(s_filename);
    int cnt = 0;
    double totac = 0;
    while(!testData.isEof()){
        cnt++;
        QString pass = "Pass " + QString::number(cnt) + "\n";
        output_text->append(pass);
        if(testData.getNextInputs(inputVals) != topology[0])
            break;
        myNet.feedForward(inputVals);

        myNet.getResults(resultVals);

        testData.getTargetOutputs(targetVals);
        if(resultVals[0] > 0.5)
            resultVals[0] = 1;
        else
            resultVals[0] = 0;
        showVectorVals("Target", targetVals,output_text);
        showVectorVals("Results:", resultVals,output_text);
        if(resultVals[0] == targetVals[0])
            totac++;
    }
    QString output = QString::fromStdString("") + "测试准确率为 ";
    output += QString::number((totac / cnt)*100) + "%";
    output_text->append(output);
    // 显示测试结果
    QMessageBox::information(this, "测试结果", output);

}

void NeuralNetworkGUI:: test() {
    // 这里是测试函数的代码，获取测试文件名字并计算准确率
    QString filename = filename_edit->text();

    // 在文本框中追加测试信息
    output_text->append("测试文件：");
    output_text->append(filename);
    output_text->append("开始测试...\n");

    // ...
}

void NeuralNetworkGUI::openFolder() {
    QString foldername = QFileDialog::getExistingDirectory(this, "打开文件夹", ".", QFileDialog::ShowDirsOnly);
    if (!foldername.isEmpty()) {
        filename_edit->setText(foldername);
    }
}

void NeuralNetworkGUI:: openFile() {
   QString filename = QFileDialog::getOpenFileName(this, "打开文件", ".", "文本文件 (*.txt)");
   if (!filename.isEmpty()) {
       filename_edit->setText(filename);
   }
}

void NeuralNetworkGUI::clearOut() {
    output_text->clear();
}

NeuralNetworkGUI::~NeuralNetworkGUI()
{
    delete ui;
}
