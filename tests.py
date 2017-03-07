import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from layers import LinearLayer, ReluLayer, SigmoidLayer, SoftMaxLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from sequential import Sequential

class LinearLayerTests(unittest.TestCase):
    def test_dim(self):
        l = LinearLayer(5,6)
        y = l.forward(np.random.rand(5))
        self.assertEqual(y.shape,(6,))

    def test_forward(self):
        l = LinearLayer(2,6,'ones')
        y = l.forward(np.array([2.0,5.0]))
        assert_array_equal(y,np.array([8.0,8.0,8.0,8.0,8.0,8.0]))

    def test_backward(self):
        l = LinearLayer(2,6,'ones')
        d = l.backward(np.array([1.0,2.0,3.0,4.0,1.0,1.0]))
        self.assertEqual(d.shape,(2,))
        assert_array_equal(d,np.array([12.0,12.0]))

    def test_update(self):
        l = LinearLayer(2,6,'ones')
        y = l.forward(np.array([2.0,2.0]))
        dW = l.update(np.array([1.0,2.0,3.0,4.0,1.0,1.0]))
        self.assertEqual(l.W.shape,(6,3))
        self.assertEqual(dW.shape,(6,3))

        l = LinearLayer(2,3,'ones')
        y = l.forward(np.array([2.0,2.0]))
        dW = l.update(np.array([1.0,2.0,3.0]))
        self.assertEqual(l.W.shape,(3,3))
        self.assertEqual(dW.shape,(3,3))
        assert_array_equal(dW,np.matrix([[2.0,2.0,1.0],[4.0,4.0,2.0],[6.0,6.0,3.0]]))

    def test_numeric_gradient(self):
        l = LinearLayer(2,3,'random')
        weights = l.W
        x = np.random.rand(2)
        grad = l.numeric_gradient(x)
        assert_almost_equal(grad,weights[:,0:-1])

        in_delta = np.random.rand(3)
        for i,d in enumerate(in_delta):
            aux = np.zeros(in_delta.size)
            aux[i] = in_delta[i]
            delta = l.backward(aux)
            gradient = l.numeric_gradient(x)
            assert_almost_equal(in_delta[i]*gradient[i,:],delta,decimal=5)

class ReluLayerTests(unittest.TestCase):
    def test_forward_backward(self):
        l = ReluLayer()
        y = l.forward(np.array([5,5]))
        self.assertEqual(y.shape,(2,))
        assert_almost_equal(y,np.array([5, 5]))
        x = np.random.rand(2)
        d = l.backward(x)
        self.assertEqual(d.shape,(2,))
        assert_almost_equal(d,np.array(x),decimal=5)
        return

    def test_numeric_gradient(self):
        l = ReluLayer()
        x = np.random.rand(2)
        gradient = l.numeric_gradient(x)
        l.forward(x)
        delta = l.backward([1,1])
        assert_almost_equal(np.diag(gradient),delta,decimal=5)


class SigmoidLayerTests(unittest.TestCase):
    def test_forward_backward(self):
        l = SigmoidLayer()
        y = l.forward(np.array([5,5]))
        self.assertEqual(y.shape,(2,))
        assert_almost_equal(y,np.array([0.993307, 0.993307]),decimal=5)
        d = l.backward(np.array([2,3]))
        self.assertEqual(d.shape,(2,))
        assert_almost_equal(d,np.array([0.0132961, 0.0199442]),decimal=5)
        return

    def test_numeric_gradient(self):
        l = SigmoidLayer()
        x = np.random.rand(2)
        gradient = l.numeric_gradient(x)
        l.forward(x)
        delta = l.backward([1,1])
        assert_almost_equal(np.diag(gradient),delta)

class SoftMaxLayerTests(unittest.TestCase):
    def test_forward_backward(self):
        l = SoftMaxLayer()
        y = l.forward(np.array([5,5,6]))
        self.assertEqual(y.shape,(3,))
        assert_almost_equal(y,np.array([ 0.2119416,  0.2119416,  0.5761169]),decimal=5)
        assert_array_equal(1,np.sum(y))
        d = l.backward(np.array([2,3,6]))
        self.assertEqual(d.shape,(3,))
        #assert_almost_equal(d,np.array([-1.792177 , -1.5802354,  1.2406412]))
        return

    def test_numeric_gradient(self):
        l = SoftMaxLayer()
        x = np.random.rand(3)
        in_delta = np.random.rand(3)
        for i,d in enumerate(in_delta):
            aux_delta = np.zeros(in_delta.size)
            aux_delta[i] = in_delta[i]
            l.forward(x)
            delta = l.backward(aux_delta)
            gradient = l.numeric_gradient(x)
            assert_almost_equal(in_delta[i]*gradient[i,:],delta,decimal=5)

class NegativeLogLikelihoodLossTests(unittest.TestCase):
    def test_calc_loss(self):
        l1 = SoftMaxLayer()
        n = Sequential([l1])
        x = np.array([15.0,10.0,2.0])
        y = n.forward(x)
        self.assertEqual(y.shape,(3,))
        nll = NegativeLogLikelihoodLoss()
        t = np.array([0.0,0.0,1.0])
        self.assertEqual(y.shape,t.shape)
        J = nll.calc_loss(y,t)
        self.assertEqual(J.shape,(3,))
        assert_almost_equal(J,[0.0,0.0,13.0067176],decimal=5)

    def test_calc_delta(self):
        l1 = SoftMaxLayer()
        n = Sequential([l1])
        x = np.array([15.0,10.0,2.0])
        y = n.forward(x)
        self.assertEqual(y.shape,(3,))
        nll = NegativeLogLikelihoodLoss()
        t = np.array([0.0,0.0,1.0])
        self.assertEqual(y.shape,t.shape)
        J = nll.calc_loss(y,t)
        self.assertEqual(J.shape,(3,))
        assert_almost_equal(J,[0.0,0.0,13.0067176],decimal=5)
        delta_in = -nll.calc_gradient(y,t)
        assert_almost_equal(delta_in,[0.0,0.0,445395.349996],decimal=5)
        delta_out = n.backward(delta_in)
        assert_almost_equal(delta_out,[-0.9933049, -0.0066928,  0.9999978],decimal=5)

    def test_numeric_gradient(self):
        nll = NegativeLogLikelihoodLoss()
        y = np.random.rand(2)
        t = np.random.rand(2)
        nll.calc_loss(y,t)
        gradient = nll.numeric_gradient(y)
        delta = nll.backward(y)
        assert_almost_equal(np.diag(gradient),delta,decimal=5)


class CrossEntropyLossTests(unittest.TestCase):
    def test_calc_loss(self):
        l1 = SoftMaxLayer()
        n = Sequential([l1])
        x = np.array([15.0,10.0,2.0])

        y = n.forward(x)
        self.assertEqual(y.shape,(3,))
        nll = NegativeLogLikelihoodLoss()
        t = np.array([0.0,0.0,1.0])
        self.assertEqual(y.shape,t.shape)
        J1 = nll.calc_loss(y,t)
        self.assertEqual(J1.shape,(3,))
        assert_almost_equal(J1,[0.0,0.0,13.0067176],decimal=5)

        cel = CrossEntropyLoss()
        t = np.array([0.0,0.0,1.0])
        J2 = cel.calc_loss(x,t)
        self.assertEqual(J2.shape,(3,))
        assert_almost_equal(J2,[0.0,0.0,13.0067176],decimal=5)

        assert_almost_equal(J1,J2)

    def test_calc_delta(self):
        l1 = SoftMaxLayer()
        n = Sequential([l1])
        x = np.array([15.0,10.0,2.0])
        y = n.forward(x)
        self.assertEqual(y.shape,(3,))
        nll = NegativeLogLikelihoodLoss()
        t = np.array([0.0,0.0,1.0])
        self.assertEqual(y.shape,t.shape)
        J1 = nll.calc_loss(y,t)
        self.assertEqual(J1.shape,(3,))
        assert_almost_equal(J1,[0.0,0.0,13.0067176],decimal=5)

        cel = CrossEntropyLoss()
        t = np.array([0.0,0.0,1.0])
        J2 = cel.calc_loss(x,t)
        self.assertEqual(J2.shape,(3,))
        assert_almost_equal(J2,[0.0,0.0,13.0067176],decimal=5)

        delta_in = -nll.calc_gradient(y,t)
        assert_almost_equal(delta_in,[0.0,0.0,445395.349996])
        delta_out1 = n.backward(delta_in)
        assert_almost_equal(delta_out1,[-0.9933049, -0.0066928,  0.9999978],decimal=5)
        #
        cel.calc_gradient(x,t)
        delta_out2 = cel.backward(x)
        assert_almost_equal(delta_out2,[-0.9933049, -0.0066928,  0.9999978],decimal=5)

    # def test_numeric_gradient(self):
    #     cel = CrossEntropyLoss()
    #     y = np.random.rand(2)
    #     t = np.random.rand(2)
    #     cel.calc_loss(y,t)
    #     gradient = cel.numeric_gradient(y)
    #     delta = cel.backward(y)
    #     assert_almost_equal(gradient,delta)

    # def test_calc_delta(self, y, t):
    #     #the gradient is positive but delta is negative
    #     return t/y

# class CrossEntropyLossTests(unittest.TestCase):
#     def calc_loss(self, y, t):
#         totlog = np.log(np.sum(np.exp(y)))
#         return t*(totlog - y)
#
#     def calc_delta(self, y, t):
#         return y-t

class SequentialTests(unittest.TestCase):
    def test_LinearLayer(self):
        l1 = LinearLayer(5,6,'ones')
        n = Sequential([l1])
        y = n.forward(np.array([2.0,1.0,2.0,3.0,4.0]))
        self.assertEqual(y.shape,(6,))
        assert_array_equal(y,np.array([13.0,13.0,13.0,13.0,13.0,13.0,]))

        l2 = LinearLayer(6,2,'ones')
        n.add(l2)
        y = n.forward(np.array([2.0,1.0,2.0,3.0,4.0]))
        self.assertEqual(y.shape,(2,))
        assert_array_equal(y,np.array([79.0,79.0]))

        d = n.backward(np.array([2.0,3.0]))
        self.assertEqual(d.shape,(5,))
        assert_array_equal(d,np.array([30.,30.,30.,30.,30.]))

    def test_SigmoidLayer(self):
        l1 = SigmoidLayer()
        n = Sequential([l1])
        y = n.forward(np.array([0]))
        self.assertEqual(y.shape,(1,))
        assert_array_equal(y,np.array([0.5]))

        d = n.backward(np.array([1]))
        self.assertEqual(d.shape,(1,))
        assert_array_equal(d,np.array([0.25]))

    # def test_SquaredLoss(self):
    #     errSq = SquaredLoss()
    #     n = Sequential([errSq])
    #     y = n.forward(np.array([0]))
    #     self.assertEqual(y.shape,(1,))
    #     assert_array_equal(y,np.array([0]))
    #
    #     d = n.backward(np.array([5.4]))
    #     self.assertEqual(d.shape,(1,))
    #     assert_array_equal(d,np.array([5.4]))



        # n.learn([2.0,-1.0,2.0,-2.0,-4.0],[0.3,0.9],errSq,0.5)
        # y = n.forward([2.0,1.0,2.0,-2.0,-4.0])
        # assert_almost_equal(y,np.array([ 0.6927011,  0.6932031]))
        #
        # n.learn([2.0,-1.0,2.0,-2.0,-4.0],[0.3,0.9],errSq,0.5)
        # y = n.forward([2.0,1.0,2.0,-2.0,-4.0])
        # assert_almost_equal(y,np.array([ 0.6611633,  0.6622684]))



    # def test_sequential_forward(self):
    #     l1 = LinearLayer(5,6,'ones')
    #     l2 = LinearLayer(6,2,'ones')
    #     n = Sequential([l1,l2])
    #     o = n.backward([2.0,1.0,2.0,3.0,4.0])
    #     self.assertEqual(o.shape,(2,))
    #     assert_array_equal(o,np.array([79.0,79.0]))

