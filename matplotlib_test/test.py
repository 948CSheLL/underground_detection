import matplotlib.pyplot as plt
import numpy as np


def test1():
    # 然后创建两组数据，使用np.linspace定义x：范围是(-3,3)，个数是50，将产生一组（-3，3）内均匀分布的50个数；
    # (x,y1)表示曲线1，(x,y2)表示曲线2
    x = np.linspace(-3, 3, 50)
    print("x type: ", type(x))
    y1 = 2 * x + 1
    y2 = x ** 2
    # 定义图像窗口并画图
    # 在画图前使用plt.figure()定义一个图像窗口：编号为3；大小为(8, 5)；这两项参数可缺省。其中，num参数决定
    # 了程序运行后弹出的图像窗口名字，但在klab平台下不会显示。接着，我们使用plt.plot画出(x ,y2)曲线；使用
    # plt.plot画(x ,y1)曲线，曲线的颜色属性(color)为红色；曲线的宽度(linewidth)为1.0；曲线的类型(linestyle)
    # 为虚线，除了虚线外，大家还可使用以下线性：'-'、'--'、'-.'、':' 。接着，我们使用plt.show()显示图像。
    plt.figure(num=3, figsize=(8, 5))
    plt.plot(x, y2)
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
    plt.show()


def test2():
    # 然后创建两组数据，使用np.linspace定义x：范围是(-3,3)，个数是50，将产生一组（-3，3）内均匀分布的50个数；
    # (x,y1)表示曲线1，(x,y2)表示曲线2
    x = np.linspace(-3, 3, 50)
    print("x type: ", type(x))
    y1 = 2 * x + 1
    y2 = x ** 2
    # 定义坐标轴名称及范围
    # 使用plt.xlim设置x坐标轴范围：(-1, 2)； 使用plt.ylim设置y坐标轴范围：(-2, 3)； 使用plt.xlabel设
    # 置x坐标轴名称：’I am x’； 使用plt.ylabel设置y坐标轴名称：’I am y’；
    plt.figure(num=3, figsize=(8, 5), )
    plt.plot(x, y2)
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
    plt.xlim((-1, 2))
    plt.ylim((-2, 3))
    plt.xlabel('I am x')
    plt.ylabel('I am y')
    plt.show()


def test3():
        # 然后创建两组数据，使用np.linspace定义x：范围是(-3,3)，个数是50，将产生一组（-3，3）内均匀分布的50个数；
    # (x,y1)表示曲线1，(x,y2)表示曲线2
    x = np.linspace(-3, 3, 50)
    print("x type: ", type(x))
    y1 = 2 * x + 1
    y2 = x ** 2
    # 定义坐标轴刻度及名称
    # 有时候，我们的坐标轴刻度可能并不是一连串的数字，而是一些文字，或者我们想要调整坐标轴的刻度的稀疏，这时，就需要使
    # 用plt.xticks()或者plt.yticks()来进行调整：首先，使用np.linspace定义新刻度范围以及个数：范围是(-1,2);个数
    # 是5。使用plt.xticks设置x轴刻度：范围是(-1,2);个数是5。使用plt.yticks设置y轴刻度以及名称：刻度为
    # [-2, -1.8, -1, 1.22, 3]；对应刻度的名称为[‘really bad’,’bad’,’normal’,’good’, ‘really good’]。
    # 使用plt.show()显示图像。
    plt.figure(num=3, figsize=(8, 5))
    plt.plot(x, y2)
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
    plt.xlim((-1, 2))
    plt.ylim((-2, 3))
    plt.xlabel('I am x')
    plt.ylabel('I am y')
    new_ticks = np.linspace(-1, 2, 5)
    print(new_ticks)
    plt.xticks(new_ticks)
    plt.yticks([-2, -1.8, -1, 1.22, 3], [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
    plt.show()


def test4():
    # 然后创建两组数据，使用np.linspace定义x：范围是(-3,3)，个数是50，将产生一组（-3，3）内均匀分布的50个数；
    # (x,y1)表示曲线1，(x,y2)表示曲线2
    x = np.linspace(-3, 3, 50)
    print("x type: ", type(x))
    y1 = 2 * x + 1
    y2 = x ** 2
    # 设置图像边框颜色
    # 细心的小伙伴可能会注意到，我们的图像坐标轴总是由上下左右四条线组成，我们也可以对它们进行修改：首先，使用
    # plt.gca()获取当前坐标轴信息。使用.spines设置边框：右侧边框；使用.set_color设置边框颜色：默认白色；
    # 使用.spines设置边框：上边框；使用.set_color设置边框颜色：默认白色；
    plt.figure(num=3, figsize=(8, 5))
    plt.plot(x, y2)
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
    plt.xlim((-1, 2))
    plt.ylim((-2, 3))
    new_ticks = np.linspace(-1, 2, 5)
    plt.xticks(new_ticks)
    plt.yticks([-2, -1.8, -1, 1.22, 3], [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.show()


def test5():
    # 然后创建两组数据，使用np.linspace定义x：范围是(-3,3)，个数是50，将产生一组（-3，3）内均匀分布的50个数；
    # (x,y1)表示曲线1，(x,y2)表示曲线2
    x = np.linspace(-3, 3, 50)
    print("x type: ", type(x))
    y1 = 2 * x + 1
    y2 = x ** 2
    # 调整刻度及边框位置
    # 使用.xaxis.set_ticks_position设置x坐标刻度数字或名称的位置：bottom.（所有位置：
    # top，bottom，both，default，none）；使用.spines设置边框：x轴；使用.set_position设置边框位置：y=0的
    # 位置；（位置所有属性：outward，axes，data）
    plt.figure(num=3, figsize=(8, 5))
    plt.plot(x, y2)
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
    plt.xlim((-1, 2))
    plt.ylim((-2, 3))
    new_ticks = np.linspace(-1, 2, 5)
    plt.xticks(new_ticks)
    plt.yticks([-2, -1.8, -1, 1.22, 3], [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    plt.show()


def test6():
    # 然后创建两组数据，使用np.linspace定义x：范围是(-3,3)，个数是50，将产生一组（-3，3）内均匀分布的50个数；
    # (x,y1)表示曲线1，(x,y2)表示曲线2
    x = np.linspace(-3, 3, 50)
    print("x type: ", type(x))
    y1 = 2 * x + 1
    y2 = x ** 2
    # 调整刻度及边框位置
    # 使用.yaxis.set_ticks_position设置y坐标刻度数字或名称的位置：left.（所有位置：left，right，both，default，none）
    # 使用.spines设置边框：y轴；使用.set_position设置边框位置：x=0的位置；（位置所有属性：outward，axes，data） 使用
    # plt.show显示图像.
    plt.figure(num=3, figsize=(8, 5))
    plt.plot(x, y2)
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
    plt.xlim((-1, 2))
    plt.ylim((-2, 3))
    new_ticks = np.linspace(-1, 2, 5)
    plt.xticks(new_ticks)
    plt.yticks([-2, -1.8, -1, 1.22, 3], [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.show()

def test7():
    # 小伙伴们，以上就是matplotlib的基本用法，是不是比较简单呢？现在，请根据上述所学内容，画出直线 y = x-1,
    # 线型为虚线，线宽为1，纵坐标范围（-2，1），横坐标范围（-1，2），横纵坐标在（0，0）坐标点相交。横坐标的
    # [-1,-0.5,1] 分别对应 [bad, normal, good]。请一定自己尝试一番再看下面的答案噢~
    # 答案
    x = np.linspace(-1, 2, 50)
    y = x - 1
    plt.figure()
    plt.plot(x, y, linewidth=1.0, linestyle='--')
    plt.xlim((-1, 2))
    plt.ylim((-2, 2))
    plt.xticks([-1, -0.5, 1], ['bad', 'normal', 'good'])
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    plt.show()


def test8():
    # matplotlib 中的 legend 图例就是为了帮我们展示出每个数据对应的图像名称.
    x = np.linspace(-3, 3, 50)
    y1 = 2*x + 1
    y2 = x**2
    plt.figure()
    # 本节中我们将对图中的两条线绘制图例，首先我们设置两条线的类型等信息（蓝色实线与红色虚线)，并且通过label
    # 参数为两条线设置名称。比如直线的名称就叫做 "linear line", 曲线的名称叫做 "square line"。当然，
    # 只是设置好名称并不能使我们的图例出现，要通过plt.legend()设置图例的显示。legend获取代码中的 label 的
    # 信息, plt 就能自动的为我们添加图例
    # set line syles
    l1 = plt.plot(x, y1, label='linear line')
    l2 = plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--', label='square line')
    plt.legend()
    plt.show()


def test9():
    # matplotlib 中的 legend 图例就是为了帮我们展示出每个数据对应的图像名称.
    x = np.linspace(-3, 3, 50)
    y1 = 2*x + 1
    y2 = x**2
    plt.figure()
    # 如果希望图例能够更加个性化，可通过以下方式更改：参数 loc 决定了图例的位置,比如参数 loc='upper right'
    # 表示图例将添加在图中的右上角。
    # 其中’loc’参数有多种，’best’表示自动分配最佳位置，其余的如下：
    # 'best' : 0,
    # 'upper right' : 1,
    # 'upper left' : 2,
    # 'lower left' : 3,
    # 'lower right' : 4,
    # 'right' : 5,
    # 'center left' : 6,
    # 'center right' : 7,
    # 'lower center' : 8,
    # 'upper center' : 9,
    # 'center' : 10
    l1 = plt.plot(x, y1, label='linear line')
    l2 = plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--', label='square line')
    plt.legend(loc='upper right')
    plt.show()


def test10():
    # matplotlib 中的 legend 图例就是为了帮我们展示出每个数据对应的图像名称.
    x = np.linspace(-3, 3, 50)
    y1 = 2*x + 1
    y2 = x**2
    plt.figure()
    # 同样可以通过设置 handles 参数来选择图例中显示的内容。首先，在上面的代码 plt.plot(x, y2, label='linear line')
    # 和 plt.plot(x, y1, label='square line') 中用变量 l1 和 l2 分别存储起来，而且需要注意的是 l1, l2,要以逗号
    # 结尾, 因为plt.plot() 返回的是一个列表。然后将 l1,l2 这样的objects以列表的形式传递给 handles。另外，label 参数可以用来
    # 单独修改之前的 label 信息, 给不同类型的线条设置图例信息。
    l1, = plt.plot(x, y1, label='linear line')
    l2, = plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--', label='square line')
    plt.legend(handles=[l1, l2, ], labels=['up', 'down'], loc='best')
    plt.show()


def test11():
    x = np.linspace(-3, 3, 50)
    y = 2 * x + 1
    plt.figure(num=1, figsize=(8, 5), )
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.plot(x, y, )
    # 然后标注出点(x0, y0)的位置信息。用plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2.5)
    # 画出一条垂直于x轴的虚线。其中，[x0, x0,], [0, y0,] 表示在图中画一条从点 (x0,y0) 到 (x0,0)
    # 的直线，'k--' 表示直线的颜色为黑色(black)，线形为虚线。而 plt.scatter 函数可以在图中画点，此
    # 时我们画的点为 (x0,y0), 点的大小(size)为 50， 点的颜色为蓝色(blue),可简写为 b。
    x0 = 1
    y0 = 2 * x0 + 1
    plt.plot([x0, x0, ], [0, y0, ], 'k--', linewidth=2.5)
    # set dot styles
    plt.scatter([x0, ], [y0, ], s=50, color='b')
    plt.show()


def test12():
    x = np.linspace(-3, 3, 50)
    y = 2 * x + 1
    plt.figure(num=1, figsize=(8, 5), )
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.plot(x, y, )
    # 然后标注出点(x0, y0)的位置信息。用plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2.5)
    # 画出一条垂直于x轴的虚线。其中，[x0, x0,], [0, y0,] 表示在图中画一条从点 (x0,y0) 到 (x0,0)
    # 的直线，'k--' 表示直线的颜色为黑色(black)，线形为虚线。而 plt.scatter 函数可以在图中画点，此
    # 时我们画的点为 (x0,y0), 点的大小(size)为 50， 点的颜色为蓝色(blue),可简写为 b。
    x0 = 1
    y0 = 2 * x0 + 1
    plt.plot([x0, x0, ], [0, y0, ], 'k--', linewidth=2.5)
    # set dot styles
    plt.scatter([x0, ], [y0, ], s=50, color='b')
    # 接下来我们就对(x0, y0)这个点进行标注。第一种方式就是利用函数 annotate()，其中 r'2x+1' %y0 代
    # 表标注的内容，可以通过字符串 %s 将 y0 的值传入字符串；参数xycoords='data' 是说基于数据的值来选位置,
    # xytext=(+30, -30) 和 textcoords='offset points' 表示对于标注位置的描述 和 xy 偏差值，即标注
    # 位置是 xy 位置向右移动 30，向下移动30, arrowprops是对图中箭头类型和箭头弧度的设置，需要用 dict 形式传入。
    plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
                 textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
    plt.show()


def test13():
    x = np.linspace(-3, 3, 50)
    y = 2 * x + 1
    plt.figure(num=1, figsize=(8, 5), )
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.plot(x, y, )
    # 然后标注出点(x0, y0)的位置信息。用plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2.5)
    # 画出一条垂直于x轴的虚线。其中，[x0, x0,], [0, y0,] 表示在图中画一条从点 (x0,y0) 到 (x0,0)
    # 的直线，'k--' 表示直线的颜色为黑色(black)，线形为虚线。而 plt.scatter 函数可以在图中画点，此
    # 时我们画的点为 (x0,y0), 点的大小(size)为 50， 点的颜色为蓝色(blue),可简写为 b。
    x0 = 1
    y0 = 2 * x0 + 1
    plt.plot([x0, x0, ], [0, y0, ], 'k--', linewidth=2.5)
    # set dot styles
    plt.scatter([x0, ], [y0, ], s=50, color='b')
    # 接下来我们就对(x0, y0)这个点进行标注。第一种方式就是利用函数 annotate()，其中 r'2x+1' %y0 代
    # 表标注的内容，可以通过字符串 %s 将 y0 的值传入字符串；参数xycoords='data' 是说基于数据的值来选位置,
    # xytext=(+30, -30) 和 textcoords='offset points' 表示对于标注位置的描述 和 xy 偏差值，即标注
    # 位置是 xy 位置向右移动 30，向下移动30, arrowprops是对图中箭头类型和箭头弧度的设置，需要用 dict 形式传入。
    plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
                 textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
    # 第二种注释方式是通过 text() 函数，其中 -3.7,3, 是选取text的位置, r'' 为 text 的内容，
    # 其中空格需要用到转字符 \ ,fontdict 设置文本字的大小和颜色。
    plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
             fontdict={'size': 16, 'color': 'r'})
    plt.show()


def test14():
    # tick 能见度
    # 当图片中的内容较多，相互遮盖时，我们可以通过设置相关内容的透明度来使图片更易于观察，
    # 也即是通过本节中的bbox参数设置来调节图像信息.首先参考之前的例子, 我们先绘制图像基本信息：
    x = np.linspace(-3, 3, 50)
    y = 0.1 * x
    plt.figure()
    # 在 plt 2.0.2 或更高的版本中, 设置 zorder 给 plot 在 z 轴方向排序
    plt.plot(x, y, linewidth=10, zorder=1)
    plt.ylim(-2, 2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    # 然后对被遮挡的图像调节相关透明度，本例中设置 x轴 和 y轴 的刻度数字进行透明度设置。其中label.set_fontsize(12)
    # 重新调节字体大小，bbox设置目的内容的透明度相关参，facecolor调节 box 前景色，edgecolor 设置边框， 本处设置边框
    # 为无，alpha设置透明度.
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
        # 在 plt 2.0.2 或更高的版本中, 设置 zorder 给 plot 在 z 轴方向排序
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7, zorder=2))
    plt.show()


def test15():
    # 散点图
    # 首先，先引入matplotlib.pyplot简写作plt,再引入模块numpy用来产生一些随机数据。
    #
    # 1.数据生成
    # 生成1024个呈标准正态分布的二维数据组 (平均数是0，方差为1) 作为一个数据集，并图像化这个数据集。每一个点的颜色值用T来表示：
    n = 1024  # data size
    X = np.random.normal(0, 1, n)  # 每一个点的X值
    Y = np.random.normal(0, 1, n)  # 每一个点的Y值
    T = np.arctan2(Y, X)  # for color value
    # 2.画图：
    # 数据集生成完毕，现在来用 plt.scatter 画出这个点集，输入X和Y作为location，size=75，颜色为T，color map用默认值，
    # 透明度alpha 为 50%。 x轴显示范围定位(-1.5，1.5)，并向xtick()函数传入空集()来隐藏x坐标轴，y轴同理：
    plt.scatter(X, Y, s=75, c=T, alpha=.5)
    plt.xlim(-1.5, 1.5)
    plt.xticks(())  # ignore xticks
    plt.ylim(-1.5, 1.5)
    plt.yticks(())  # ignore yticks
    plt.show()


def test16():
    # 柱状图
    # 柱状图是在数据分析过程中最为常用的图表，折线图和饼图能够表达的信息，柱状图都能够表达。在学术报告或工作场景下，大家应尽量使用柱状图来代替折线图与饼图。下面，我们就开始吧~
    #
    # 1.数据生成：
    # 首先生成画图数据，向上向下分别生成2组数据，X为0到11的整数 ，Y是相应的均匀分布的随机数据。
    #
    # 2.画图：
    # 使用的函数是plt.bar，参数为X和Y，X代表横坐标，即柱形的位置，Y代表纵坐标，即柱形的高度。
    n = 12
    X = np.arange(n)
    Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    plt.bar(X, +Y1)
    plt.bar(X, -Y2)
    plt.xlim(-.5, n)
    plt.xticks(())
    plt.ylim(-1.25, 1.25)
    plt.yticks(())
    plt.show()


def test17():
    # 柱状图
    # 柱状图是在数据分析过程中最为常用的图表，折线图和饼图能够表达的信息，柱状图都能够表达。在学术报告或工作场景下，大家应尽量使用柱状图来代替折线图与饼图。下面，我们就开始吧~
    #
    # 1.数据生成：
    # 首先生成画图数据，向上向下分别生成2组数据，X为0到11的整数 ，Y是相应的均匀分布的随机数据。
    #
    # 2.画图：
    # 使用的函数是plt.bar，参数为X和Y，X代表横坐标，即柱形的位置，Y代表纵坐标，即柱形的高度。
    n = 12
    X = np.arange(n)
    Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    plt.bar(X, +Y1)
    plt.bar(X, -Y2)
    plt.xlim(-.5, n)
    plt.xticks(())
    plt.ylim(-1.25, 1.25)
    plt.yticks(())
    # 3.修改颜色和数据标签
    # 如果小伙伴们想要改变柱状图的颜色，并且希望每个柱形上方能够显示该项数值该怎么做呢？我们可
    # 以用 plt.bar 函数中的facecolor参数设置柱状图主体颜色，用edgecolor参数设置边框颜色；
    # 而函数 plt.text 可以帮助我们在柱体上方（下方）加上数值：用%.2f保留两位小数，用ha='center'
    # 设置横向居中对齐，用va='bottom'设置纵向底部（顶部）对齐。
    plt.bar(X, +Y1, facecolor='#FFCCCC', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#6699CC', edgecolor='white')
    for x, y in zip(X, Y1):
        # ha: horizontal alignment
        # va: vertical alignment
        plt.text(x, y, '%.2f' % y, ha='center', va='bottom')
    for x, y in zip(X, Y2):
        # ha: horizontal alignment
        # va: vertical alignment
        plt.text(x, -y, '%.2f' % y, ha='center', va='top')
    plt.show()


if __name__ == "__main__":
    test9()