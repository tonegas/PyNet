import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from computationalgraph import Input, Weight, MatrixWeight, Sigmoid
from layers import ComputationalGraphLayer

from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from network import Sequential


from layers import WeightLayer, SumLayer, MulLayer, ConstantLayer, SigmoidLayer, LinearLayer, WeightMatrixLayer
from network import ParallelGroup, Sequential, MapGroup, SumGroup, MulGroup
from genericlayer import GenericLayer, WithElements


class ComputationGraphTests(unittest.TestCase):
    def test_one_input(self):
        x = Input(['x'],'x')
        net = ComputationalGraphLayer(x*x*3.0+x*4.3+3.1-1-x*(x**3+x*1.2+2)+1)
        xv = np.array([1.3])
        out = net.forward(xv)
        self.assertEqual(out.shape,(1,))
        assert_almost_equal(out,np.array(xv*xv*3+xv*4.3+3.1-1-xv*(xv**3+xv*1.2+2)+1))
        dJdx = net.backward(np.array([1.0]))
        self.assertEqual(dJdx.shape,(1,))
        assert_almost_equal(dJdx,-4*(-0.575 - 0.9*xv + xv**3))
        xv = np.array([1.1,2.0,3.0,4.0])
        out = net.forward(xv)
        self.assertEqual(out.shape,(4,))
        assert_almost_equal(out,np.array(xv*xv*3+xv*4.3+3.1-1-xv*(xv**3+xv*1.2+2)+1))
        dJdx = net.backward(np.array([1.0,1.0,1.0,1.0]))
        self.assertEqual(dJdx.shape,(4,))
        assert_almost_equal(dJdx,-4*(-0.575 - 0.9*xv + xv**3))

    def test_multi_input(self):
        list_var = ['x','y']#Here the order is important to understand the order of variable in the input group
        x = Input(list_var,'x')
        y = Input(list_var,'y')
        xv = np.array([1.3])
        yv = np.array([2.3])
        xyv = [xv,yv]
        net = ComputationalGraphLayer(y*x+x*x*3.0+y*y*4.3+3.1-1-x*(y**3+x*1.2+2)+1)
        out = net.forward(xyv)
        self.assertEqual(out.shape,(1,))
        assert_almost_equal(out,yv*xv+xv*xv*3.0+yv*yv*4.3+3.1-1-xv*(yv**3+xv*1.2+2)+1)
        dJdy = net.backward(np.array([1.0]))
        self.assertEqual(len(dJdy),2)
        for element in dJdy:
            self.assertEqual(element.shape,(1,))
        assert_almost_equal(dJdy,[-2+3.6*xv+yv-yv**3,xv+8.6*yv-3*xv*yv**2])

        xv = np.array([1.3,1.1,2.3,1.2])
        yv = np.array([2.3,1.2,3.1,2.2])
        xyv = [xv,yv]
        net = ComputationalGraphLayer(y*x+x*x*3.0+y*y*4.3+3.1-1-x*(y**3+x*1.2+2)+1)
        out = net.forward(xyv)
        self.assertEqual(out.shape,(4,))
        assert_almost_equal(out,yv*xv+xv*xv*3.0+yv*yv*4.3+3.1-1-xv*(yv**3+xv*1.2+2)+1)
        dJdy = net.backward(np.array([1.0]))
        self.assertEqual(len(dJdy),2)
        for element in dJdy:
            self.assertEqual(element.shape,(4,))
        assert_almost_equal(dJdy,[-2+3.6*xv+yv-yv**3,xv+8.6*yv-3*xv*yv**2])

    def test_variable_weight_one_input(self):
        #function x**2*a+x*b+c
        xv = np.array([1.3])
        x = Input(['x'],'x')
        av = np.array([2.1])
        bv = np.array([3.2])
        cv = np.array([5.1])
        a = Weight(1, weights = av)
        b = Weight(1, weights = bv)
        c = Weight(1, weights = cv)
        net = ComputationalGraphLayer(a*x**2+b*x+c)
        out = net.forward(xv)
        self.assertEqual(out.shape,(1,))
        assert_almost_equal(out,xv**2*av+xv*bv+cv)
        dJdy = net.backward(np.array([1.0]))
        self.assertEqual(dJdy.shape,(1,))
        assert_almost_equal(dJdy,2*xv*av+bv)

        net = ComputationalGraphLayer(x*x*a+x*b+c)
        out = net.forward(xv)
        self.assertEqual(out.shape,(1,))
        assert_almost_equal(out,xv**2*av+xv*bv+cv)
        dJdy = net.backward(np.array([1.0]))
        self.assertEqual(dJdy.shape,(1,))
        assert_almost_equal(dJdy,2*xv*av+bv)

    def test_variable_weight_multi_input(self):
        pass

    def test_variable_matrixweight_one_input(self):
        #function W*x vettoriale
        xv = np.array([1.3,1.1,7.5])
        x = Input(['x'],'x')
        Wv = np.array([[2.1,3.1,2.2],[2.2,3.2,4.2]])
        W = MatrixWeight(3, 2, weights = Wv)
        net = ComputationalGraphLayer(W*x)
        out = net.forward(xv)
        self.assertEqual(out.shape,(2,))
        assert_almost_equal(out,Wv.dot(xv))
        dJdy = net.backward(np.array([1.0,1.0]))
        self.assertEqual(dJdy.shape,(3,))
        assert_almost_equal(dJdy,np.sum(Wv,0))

        Wv1 = np.array([[2.1,3.1,2.2],[2.2,3.2,4.2]])
        Wv2 = np.array([[2.1,3.1],[7.4,2.2],[3.2,2.2],[1.1,1.2]])
        Wv3 = np.array([[2.1,3.1,7.4,2.2],[3.2,2.2,1.1,1.2],[2.2,3.2,4.2,7.4]])
        W1 = MatrixWeight(3, 2, weights = Wv1)
        W2 = MatrixWeight(2, 4, weights = Wv2)
        W3 = MatrixWeight(4, 3, weights = Wv3)
        net = ComputationalGraphLayer(W3*W2*W1*x)
        out = net.forward(xv)
        self.assertEqual(out.shape,(3,))
        assert_almost_equal(out,Wv3.dot(Wv2).dot(Wv1).dot(xv))
        dJdy = net.backward(np.array([1.0,1.0,1.0]))
        self.assertEqual(dJdy.shape,(3,))
        assert_almost_equal(dJdy,np.sum(Wv3.dot(Wv2).dot(Wv1),0))

        net = ComputationalGraphLayer(W3*(W2*(W1*x)))
        out = net.forward(xv)
        self.assertEqual(out.shape,(3,))
        assert_almost_equal(out,Wv3.dot(Wv2.dot(Wv1.dot(xv))))
        dJdy = net.backward(np.array([1.0,1.0,1.0]))
        self.assertEqual(dJdy.shape,(3,))
        assert_almost_equal(dJdy,np.sum(Wv3.dot(Wv2).dot(Wv1),0))

    def test_variable_mw_and_w_one_input(self):
        #test one layer W*x+b
        xv = np.array([1.5,1.1,7.5])
        x = Input(['x'],'x')
        Wv = np.array([[2.1,3.1,2.2],[2.2,3.2,4.2]])
        W = MatrixWeight(3, 2, weights = Wv)
        bv = np.array([1.3,5.1])
        b = Weight(2, weights = bv)
        net = ComputationalGraphLayer(W*x+b)
        out = net.forward(xv)
        self.assertEqual(out.shape,(2,))
        assert_almost_equal(out,Wv.dot(xv)+bv)
        dJdy = net.backward(np.array([1.0,1.0]))
        self.assertEqual(dJdy.shape,(3,))
        assert_almost_equal(dJdy,np.sum(Wv,0))

    def test_variable_mw_and_w_multi_input(self):
        #test one layer W*x+b+W*y+b
        list_var = ['x','y']
        x = Input(list_var,'x')
        y = Input(list_var,'y')
        xv = np.array([1.5,1.1,7.5])
        yv = np.array([3.5,3.1,2.5,4.3])
        xyv = [xv,yv]

        Wv1 = np.array([[2.1,3.1,2.2],[2.2,3.2,4.2]])
        Wv2 = np.array([[6.1,5.1,2.2,4.3],[1.2,1.2,5.2,5.1]])
        W1 = MatrixWeight(3, 2, weights = Wv1)
        W2 = MatrixWeight(4, 2, weights = Wv2)
        bv = np.array([1.3,5.1])
        b = Weight(2, weights = bv)
        net = ComputationalGraphLayer(W1*x+b+W2*y-b)
        out = net.forward(xyv)
        self.assertEqual(out.shape,(2,))
        assert_almost_equal(out,Wv1.dot(xv)+bv+Wv2.dot(yv)-bv)
        dJdy = net.backward(np.array([1.0,1.0]))
        self.assertEqual(len(dJdy),2)
        gradvett = [np.sum(Wv1,0),np.sum(Wv2,0)]
        for ind,element in enumerate(dJdy):
            self.assertEqual(element.shape,xyv[ind].shape)
            assert_almost_equal(dJdy[ind],gradvett[ind])

    def test_neuron_one_input(self):
        xv = np.array([0.5,0.1,0.5])
        x = Input(['x'],'x')
        Wv = np.array([[0.1,0.1,0.2],[0.5,0.2,0.2]])
        W = MatrixWeight(3, 2, weights = Wv)
        bv = np.array([0.3,0.1])
        b = Weight(2, weights = bv)
        net = ComputationalGraphLayer(Sigmoid(W*x+b))
        out = net.forward(xv)
        self.assertEqual(out.shape,(2,))
        check_out = 1.0/(1.0+np.exp(-Wv.dot(xv)-bv))
        assert_almost_equal(out,check_out)
        dJdy = net.backward(np.array([1.0,1.0]))
        self.assertEqual(dJdy.shape,(3,))
        assert_almost_equal(dJdy,np.sum(net.numeric_gradient(xv),0))
        assert_almost_equal(dJdy,(check_out*(1-check_out)).dot(Wv))

        net2 = Sequential(
            LinearLayer(3,2, weights = np.hstack([Wv,bv.reshape(2,1)])),
            SigmoidLayer
        )
        out2 = net2.forward(xv)
        assert_almost_equal(out,out2)
        dJdy2 = net.backward(np.array([1.0,1.0]))
        assert_almost_equal(dJdy,dJdy2)


