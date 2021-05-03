from federated_learning.aggregation_server import AggregationServer
from federated_learning.data_scientist import DataScientist

if __name__ == '__main__':
    data_scientist = DataScientist()
    agg_server = AggregationServer(data_scientist.model)
