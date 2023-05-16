#ifndef ALL_CLASS_H
#define ALL_CLASS_H


#include<bits/stdc++.h>

class TrainingData{
public:
    TrainingData(const std::string filename);
    bool isEof(void) {
        return m_trainingDataFile.eof();
    }
    void getTopology(std::vector<unsigned> &topology);
    unsigned getNextInputs(std::vector<double> &inputVals);
    unsigned getTargetOutputs(std::vector<double> &targetOutputVals);
private:
    std::ifstream m_trainingDataFile;
};



struct Connection {double weight, deltaWeight;};
class Neuron;
typedef std::vector<Neuron> Layer;
class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer, int layernum, int totalLayer);
    void calcOutputGradients(double targetVals);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double sigmoid(double x);
    static double sigmoidDerivative(double x);
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    std::vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

#define eta 0.15
#define alpha 0.5


// ****************** class Net ******************
class Net{
public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
    double getRecentAverageloss(void) const { return m_recentAverageloss; }
private:
    std::vector<Layer> m_layers;
    double m_loss;
    double m_recentAverageloss;
    static double m_recentAverageSmoothingFactor;
};


class TestData{
public:
    TestData(const std::string filename);
    bool isEof(void) { return testfile.eof(); }
    unsigned getNextInputs(std::vector<double> &inputVals);
    unsigned getTargetOutputs(std::vector<double> &targetOutputVals);
    std::ifstream testfile;
};




#endif // ALL_CLASS_H
