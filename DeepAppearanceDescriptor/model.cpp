#include "model.h"
#include <algorithm>

const float kRatio=0.5;
enum DETECTBOX_IDX {IDX_X = 0, IDX_Y, IDX_W, IDX_H };

DETECTBOX DETECTION_ROW::to_xyah() const
{//(centerx, centery, ration, h)
	DETECTBOX ret = tlwh;
	ret(0,IDX_X) += (ret(0, IDX_W)*kRatio);
	ret(0, IDX_Y) += (ret(0, IDX_H)*kRatio);
	ret(0, IDX_W) /= ret(0, IDX_H);
	return ret;
}

DETECTBOX DETECTION_ROW::to_tlbr() const
{//(x,y,xx,yy)
	DETECTBOX ret = tlwh;
	ret(0, IDX_X) += ret(0, IDX_W);
	ret(0, IDX_Y) += ret(0, IDX_H);
	return ret;
}

