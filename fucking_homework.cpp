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

void showVectorVals(std::string label, std::vector<double> &v){
	std::cout << label << " ";
	unsigned size = v.size();
	for(unsigned i = 0; i < size; i++)
		std::cout << v[i] << " ";
	std::cout << '\n';
}

class TestData{
public:
	TestData(const std::string filename);
	bool isEof(void) { return testfile.eof(); }
	unsigned getNextInputs(std::vector<double> &inputVals);
	unsigned getTargetOutputs(std::vector<double> &targetOutputVals);
	std::ifstream testfile;
};

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

int main(){
	TrainingData trainData("trainingData.txt");
	std::vector<unsigned> topology;

	trainData.getTopology(topology);
	Net myNet(topology);

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	while(!trainData.isEof()){
		++trainingPass;
		std::cout << '\n' << "Pass" << trainingPass;
		if(trainData.getNextInputs(inputVals) != topology[0])
			break;
		showVectorVals(": Inputs :", inputVals);
		myNet.feedForward(inputVals);
		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		std::cout << "Net recent average loss: "
		     << myNet.getRecentAverageloss() << '\n';
	}

	std::cout << '\n' << "Done" << '\n';

	TestData testData("testData.txt");
	int cnt = 0;
	double totac = 0;
	while(!testData.isEof()){
		cnt++;
		std::cout << cnt << '\n';
		if(testData.getNextInputs(inputVals) != topology[0])
			break;
		myNet.feedForward(inputVals);

		myNet.getResults(resultVals);

		testData.getTargetOutputs(targetVals);
		if(resultVals[0] > 0.5)
			resultVals[0] = 1;
		else 
			resultVals[0] = 0;
		showVectorVals("Target", targetVals);
		showVectorVals("Results:", resultVals);
		if(resultVals[0] == targetVals[0])
			totac++;
	}
	std::cout << "Accuracy: " << totac / cnt << '\n';
}