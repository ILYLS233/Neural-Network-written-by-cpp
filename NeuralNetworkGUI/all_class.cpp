#include "all_class.h"

void TrainingData::getTopology(std::vector<unsigned> &topology){
    std::string line, label;

    getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    ss >> label;
    if(this->isEof() || label.compare("topology:") != 0)
        abort();
    while(!ss.eof()){
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    return;
}

TrainingData::TrainingData(const std::string filename){
    m_trainingDataFile.open(filename.c_str());
}


unsigned TrainingData::getNextInputs(std::vector<double> &inputVals){
    inputVals.clear();
    std::string line, label;
    getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue)
            inputVals.push_back(oneValue);
    }
    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(std::vector<double> &targetOutputVals){
    targetOutputVals.clear();
    std::string line, label;
    getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue)
            targetOutputVals.push_back(oneValue);
    }
    return targetOutputVals.size();
}

void Neuron::updateInputWeights(Layer &prevLayer){
    unsigned size = prevLayer.size();
    for(unsigned i = 0; i < size; i++){
        Neuron &neuron = prevLayer[i];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0.0;
    unsigned size = nextLayer.size();
    for (unsigned i = 0; i < size - 1; i++)
        sum += m_outputWeights[i].weight * nextLayer[i].m_gradient;
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sigmoid(double x){return 1.0 / (1.0 + exp(-x));}
double Neuron::sigmoidDerivative(double x){return x * (1.0 - x);}

void Neuron::calcOutputGradients(double targetVals){
    double delta = targetVals - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x){return sigmoid(x);}
double Neuron::transferFunctionDerivative(double x){return sigmoidDerivative(x);}

void Neuron::feedForward(const Layer &prevLayer, int layernum, int totalLayer){
    double sum = 0.0;
    for(unsigned i = 0 ; i < prevLayer.size(); i++){
        sum += prevLayer[i].getOutputVal() *
                 prevLayer[i].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for(unsigned c = 0; c < numOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

double Net::m_recentAverageSmoothingFactor = 100.0;
void Net::getResults(std::vector<double> &resultVals) const{
    resultVals.clear();
    unsigned size = m_layers.back().size();
    for(unsigned i = 0; i < size - 1; i++)
        resultVals.push_back(m_layers.back()[i].getOutputVal());
}

void Net::backProp(const std::vector<double> &targetVals){
    Layer &outputLayer = m_layers.back();
    m_loss = 0.0;
    unsigned size = outputLayer.size();
    for(unsigned i = 0; i < size - 1; i++){
        double delta = targetVals[i] - outputLayer[i].getOutputVal();
        m_loss += delta *delta;
    }
    m_loss /= outputLayer.size() - 1;
    m_loss = sqrt(m_loss);
    m_recentAverageloss =
            (m_recentAverageloss * m_recentAverageSmoothingFactor + m_loss)
            / (m_recentAverageSmoothingFactor + 1.0);
    for(unsigned i = 0; i < size - 1; i++)
        outputLayer[i].calcOutputGradients(targetVals[i]);
    for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--){
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
        unsigned size = hiddenLayer.size();
        for(unsigned i = 0; i < size; i++)
            hiddenLayer[i].calcHiddenGradients(nextLayer);
    }

    for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];
        unsigned size = layer.size();
        for(unsigned i = 0; i < size - 1; i++)
            layer[i].updateInputWeights(prevLayer);
    }
}

void Net::feedForward(const std::vector<double> &inputVals){
    assert(inputVals.size() == m_layers[0].size() - 1);
    unsigned size = inputVals.size();
    for(unsigned i = 0; i < size; i++)
        m_layers[0][i].setOutputVal(inputVals[i]);
    for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
        Layer &prevLayer = m_layers[layerNum - 1];
        unsigned size = m_layers[layerNum].size();
        for(unsigned i = 0; i < size - 1; i++){
            m_layers[layerNum][i].feedForward(prevLayer, layerNum, m_layers.size() - 1);
        }
    }
}

Net::Net(const std::vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 :topology[layerNum + 1];
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            std::cout << "Mad a Neuron!" << '\n';
        }
        m_layers.back().back().setOutputVal(1.0);
    }
}



TestData::TestData(const std::string filename){
    testfile.open(filename.c_str());
}

unsigned TestData::getNextInputs(std::vector<double> &inputVals){
    inputVals.clear();
    std::string line, label;
    getline(testfile, line);
    std::stringstream ss(line);
    ss >> label;
    if(label.compare("in:") == 0){
        double oneValue;
        while(ss >> oneValue)
            inputVals.push_back(oneValue);
    }
    return inputVals.size();
}

unsigned TestData::getTargetOutputs(std::vector<double> &targetOutputVals){
    targetOutputVals.clear();
    std::string line, label;
    getline(testfile, line);
    std::stringstream ss(line);
    ss >> label;
    if(label.compare("out:") == 0){
        double oneValue;
        while(ss >> oneValue)
            targetOutputVals.push_back(oneValue);
    }
    return targetOutputVals.size();
}
