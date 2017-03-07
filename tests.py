import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from framework import LinearLayer, ReluLayer, SigmoidLayer, SoftMaxLayer, SquaredLoss, Sequential

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
            assert_almost_equal(in_delta[i]*gradient[i,:],delta)

class ReluLayerTests(unittest.TestCase):
    def test_forward_backward(self):
        l = ReluLayer()
        y = l.forward(np.array([5,5]))
        self.assertEqual(y.shape,(2,))
        assert_almost_equal(y,np.array([5, 5]))
        x = np.random.rand(2)
        d = l.backward(x)
        self.assertEqual(d.shape,(2,))
        assert_almost_equal(d,np.array(x))
        return

    def test_numeric_gradient(self):
        l = ReluLayer()
        x = np.random.rand(2)
        gradient = l.numeric_gradient(x)
        l.forward(x)
        delta = l.backward([1,1])
        assert_almost_equal(np.diag(gradient),delta)


class SigmoidLayerTests(unittest.TestCase):
    def test_forward_backward(self):
        l = SigmoidLayer()
        y = l.forward(np.array([5,5]))
        self.assertEqual(y.shape,(2,))
        assert_almost_equal(y,np.array([0.993307, 0.993307]))
        d = l.backward(np.array([2,3]))
        self.assertEqual(d.shape,(2,))
        assert_almost_equal(d,np.array([0.0132961, 0.0199442]))
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
        assert_almost_equal(y,np.array([ 0.2119416,  0.2119416,  0.5761169]))
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
            assert_almost_equal(in_delta[i]*gradient[i,:],delta)

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

    def test_SquaredLoss(self):
        errSq = SquaredLoss()
        n = Sequential([errSq])
        y = n.forward(np.array([0]))
        self.assertEqual(y.shape,(1,))
        assert_array_equal(y,np.array([0]))

        d = n.backward(np.array([5.4]))
        self.assertEqual(d.shape,(1,))
        assert_array_equal(d,np.array([5.4]))



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

