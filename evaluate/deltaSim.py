from bartGetRes import generate_bart
from t5GetRes import t5demo
from FKGL import fkgl
import time as time

if __name__ == '__main__':
    start = time.time()
    sentence = "The spacecraft consists of two main elements: the NASA Cassini orbiter, named after the Italian-French astronomer Giovanni Domenico Cassini, and the ESA Huygens probe, named after the Dutch astronomer, mathematician and physicist Christiaan Huygens."
    t5Res = t5demo.t5Generate(sentence)
    bartRes = generate_bart.bartGenerate(sentence)
    t5score = fkgl.getFKGL(t5Res)
    bartscore = fkgl.getFKGL(bartRes)
    print(t5Res)
    print(t5score)
    print(bartRes)
    print(bartscore)
    print(time.time()-start)