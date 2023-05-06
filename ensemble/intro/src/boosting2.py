import numpy as np
import support
from pruning import reducederror, alphabeta
from weighted import w_gini, WeighedZeroRule, WeighedDecisionTree
from weightedreg import w_deviation, WeighedLinear

class AdaBoostM1:
	def __init__( self, boost=10, max_depth=2, use_pruning=False ):
		self.boost = boost
		self.max_depth = max_depth
		self.use_pruning = use_pruning
		self.trees = None
		self.beta = None
		self.n_clz = 0
		
	def fit( self, x, y ):
		self.trees = []
		self.beta = np.zeros( ( self.boost, ) )
		self.n_clz = y.shape[ 1 ]
		weights = np.ones( ( len( x ), ) ) / len( x )
		for i in range( self.boost ):
			prun = None
			if self.use_pruning:
				prun = reducederror
			tree = WeighedDecisionTree( prunfnc=prun, max_depth=self.max_depth, metric=w_gini, leaf=WeighedZeroRule )
			tree.fit( x, y, weights )
			z = tree.predict( x )
			filter = z.argmax( axis=1 ) == y.argmax( axis=1 )
			err = weights[ filter==False ].sum()
			print( 'itre #%d -- error=%f'%( i+1, err ) )
			if i == 0 and err == 0:
				self.trees.append( tree )
				self.beta = self.beta[ :i+1 ]
				break
			if err >= 0.5 or err == 0:
				self.beta = self.beta[ :i ]
				break
			self.trees.append( tree )
			self.beta[ i ] = err / ( 1.0 - err )
			weights[ filter ] *= self.beta[ i ]
			weights /= weights.sum()
	
	def predict( self, x ):
		z = np.zeros( ( len(x), self.n_clz ) )
		w = np.log( 1.0 / self.beta )
		for i, tree in enumerate( self.trees ):
			p = tree.predict( x )
			c = p.argmax( axis=1 )
			for j in range(len(x)):
				z[ j, c[ j ] ] += w[ i ]
		return z
		
	def __str__( self ):
		s = []
		for i, t in enumerate( self.trees ):
			s.append( 'tree: #%s -- weight=%f'%( i+1, np.log( 1.0 / self.beta[i] ) ) )
			s.append( str( t ) )
		return '\n'.join( s )


class AdaBoostRT:
	def __init__( self, threshold=0.5, boost=10, max_depth=2, use_pruning=False ):
		self.boost = boost
		self.max_depth = max_depth
		self.use_pruning = use_pruning
		self.trees = None
		self.beta = None
		self.threshold = threshold
		
	def fit( self, x, y ):
		self.trees = []
		self.beta = np.zeros( ( self.boost, ) )
		weights = np.ones( ( len( x ), ) ) / len( x )
		threshold = self.threshold
		y_range = np.max( y ) - np.min( y )
		for i in range( self.boost ):
			prun = None
			if self.use_pruning:
				prun = alphabeta
			tree = WeighedDecisionTree( prunfnc=prun, max_depth=self.max_depth, metric=w_deviation, leaf=WeighedLinear )
			tree.fit( x, y, weights )
			z = tree.predict( x )
			l = np.absolute( z - y ) / y_range
			filter = l > threshold
			err = weights[ filter==False ].sum()
			print( 'itre #%d -- error=%f'%( i+1, err ) )
			if i == 0 and err == 0:
				self.trees.append( tree )
				self.beta = self.beta[ :i+1 ]
				break
			self.trees.append( tree )
			self.beta[ i ] = np.power( err, 2 )
			weights *= [ 1 if f else self.beta[ i ] for f in filter ]
			weights /= weights.sum()
	
	def predict( self, x ):
		z = np.zeros( ( len(x), 1 ) )
		w = np.log( 1.0 / self.beta )
		w /= w.sum()
		for i, tree in enumerate( self.trees ):
			p = tree.predict( x )
			z += p * w[ i ]
		return z


class AdaBoostR2:
	def __init__( self, boost=10, max_depth=2, use_pruning=False ):
		self.boost = boost
		self.max_depth = max_depth
		self.use_pruning = use_pruning
		self.trees = None
		self.beta = None
		
	def fit( self, x, y ):
		self.trees = []
		self.beta = np.zeros( ( self.boost, ) )
		weights = np.ones( ( len( x ), ) ) / len( x )
		for i in range( self.boost ):
			prun = None
			if self.use_pruning:
				prun = alphabeta
			tree = WeighedDecisionTree( prunfnc=prun, max_depth=self.max_depth, metric=w_deviation, leaf=WeighedLinear )
			tree.fit( x, y, weights )
			z = tree.predict( x )
			# 差分
			l = np.absolute( z[ :,0 ] - y[ :,0 ] )
			den = np.max( l )
			if def > 0:
				loss = l / den  # 最大の差が1になるようにする
			# 最大の値が1になるようにしたので、ランダムな出力だと期待値が1/2になる
			# 重み付けをして合算し、1/2より小さな間だけ、ブースティングを繰り返す
			err = np.sum( weights * loss )
			print( 'itre #%d -- error=%f'%( i+1, err ) )
			if i == 0 and err == 0:
				self.trees.append( tree )
				self.beta = self.beta[ :i+1 ]
				break
			if err >= 0.5 or err == 0:
				self.beta = self.beta[ :i ]
				break
			self.trees.append( tree )
			"""
			 貢献度の合算 = Σlog( 1 / β ) = II1/β = IIEt/II(1-Et) = 全モデルのbetaの掛け算の逆数
			 = 全モデルのエラーの掛け算/全モデルの(1-エラー) = 1/2より小さな値ならば精度が向上
			"""
			self.beta[ i ] = err / ( 1.0 - err )
			weights *= [ np.power( self.beta[ i ], 1.0 - lo ) for lo in loss ]
			weights /= weights.sum()
	
	def predict( self, x ):
		# 各モデルの貢献度を求める
		w = np.log( 1.0 / self.beta )
		# 各モデルの実行結果を予め求めておく
		pred = [ tree.predict( x ).reshape( ( -1, ) ) for tree in self.trees ]
		pred = np.array( pred ).T  # 対角にするので(データの個数×モデルの数)になる
		"""
		 モデルの出力の値が小さい順番で貢献度を合算してゆき、
		 全ての貢献度の合計の1/2以上にならない下界を求める（inf=下界を求める記号）
		 出力が小さい方から並べて合算することで、可能な限り多くのモデルを使用できる
		 つまり貢献度の合計が1/2より小さい値で出来るだけ沢山のモデルを集める
		 プログラム的には予め全てを求めておいてから、下界となる条件を検索する
		"""
		# まずそれぞれのモデルの出力を、小さい順に並べて累計する（最終的には合算になるから）
		idx = np.argsort( pred, axis=1 )  # 小さい順番の並び
		cdf = w[ idx ].cumsum( axis=1 )  # 貢献度を並び順に累計してゆく
		cbf_last = cdf[ :,-1 ].reshape((-1,1))  # 累計の最後から合計を取得して整形
		# 下界を求める〜プログラム上は全部計算する
		above = cdf >= ( 1 / 2 ) * cbf_last   # これはTrueとFalseの二次元配列になる
		# 下界となる場所のインデックスを探す〜
		median_idx = above.argmax(axis=1)   # Trueが最初に現れる位置
		# そのインデックスにある出力の場所（最初に並べ替えたから）
		median_estimators = idx[ np.arange( len(x) ), median_idx ]
		# その出力の場所にある実行結果の値を求めて返す
		return pred[ np.arange( len(x) ), median_estimators ]
		
	def __str__( self ):
		s = []
		for i, t in enumerate( self.trees ):
			s.append( 'tree: #%s -- weight=%f'%( i+1, np.log( 1.0 / self.beta[i] ) ) )
			s.append( str( t ) )
		return '\n'.join( s )


if __name__ == '__main__':
	import pandas as pd
	ps = support.get_base_args()
	ps.add_argument( '--boost', '-b', type=int, default=10, help='Bum of Boosting' )
	ps.add_argument( '--depth', '-d', type=int, default=2, help='Max Tree Depth' )
	ps.add_argument( '--pruning', '-p', action='store_true', help='Use pruning' )
	args = ps.parse_args()

	df = pd.read_csv( args.input, sep=args.separator, header=args.header, index_col=args.indexcol )
	x = df[ df.columns[ :-1 ] ].values
	
	if not args.regression:
		y, clz = support.clz_to_prob( df[ df.columns[ -1 ] ] )	
		plf = AdaBoostM1( boost=args.boost, max_depth=args.depth, use_pruning=args.pruning )
		support.report_classifier( plf, x, y, clz, args.crossvalidate )
	else:
		y = df[ df.columns[ -1 ] ].values.reshape( ( -1, 1 ) )
		plf = AdaBoostRT( boost=args.boost, max_depth=args.depth, use_pruning=args.pruning )
		support.report_regressor( plf, x, y, args.crossvalidate )
		

