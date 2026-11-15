import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, sys
O=sys.argv[1]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42})
cc=pd.read_csv(f"{O}/table_concordance.csv").set_index("frontal_rule")
mz=pd.read_csv(f"{O}/table_missed_by_zshare.csv"); mz=mz[mz.by=="zt_tert"]
rc=pd.read_csv(f"{O}/table_roc.csv").set_index("comparison")
B,G="#1f5fbf","#9a9a9a"
fig=plt.figure(figsize=(13.28/2.54*1.5,5.31/2.54*1.5))
fig.text(0.5,0.94,"Spatial vs frontal QRS-T angle in 18,558 ECGs (PTB-XL+)",ha="center",fontsize=11,weight="bold")
ax1=fig.add_axes([0.06,0.18,0.24,0.6]); ax2=fig.add_axes([0.40,0.18,0.24,0.6]); ax3=fig.add_axes([0.74,0.18,0.24,0.6])
v=[cc.loc["fr_abn_90"].pct_spatial_abn_missed_by_frontal, cc.loc["fr_abn_ref"].pct_spatial_abn_missed_by_frontal]
ax1.bar([0,1],v,color=[G,B],width=0.6)
for i,y in enumerate(v): ax1.text(i,y+1.5,f"{y:.0f}%",ha="center",fontsize=10,weight="bold")
ax1.set_xticks([0,1],["Frontal\n≤ 90°","Frontal within\nreference limit"]); ax1.set_ylim(0,50)
ax1.set_title("Spatial abnormalities\nmissed by the frontal angle",fontsize=9)
y=mz.missed_pct_ref.values
ax2.plot([0,1,2],y,"-o",color=B,lw=2)
for i,t in enumerate(y): ax2.text(i,t+2,f"{t:.0f}%",ha="center",fontsize=10,weight="bold")
ax2.set_xticks([0,1,2],["Low","Mid","High"]); ax2.set_ylim(0,50); ax2.set_xlim(-0.4,2.4)
ax2.set_xlabel("Out-of-plane share of T vector (tertile)",fontsize=8)
ax2.set_title("Misses rise when the T vector\nleaves the frontal plane",fontsize=9)
cats=["Any pathology","MI","CD"]; lab=["Any\nabnormality","Infarction","Conduction\ndisturbance"]
import numpy as np; x=np.arange(3)
ax3.bar(x-0.18,[rc.loc[c].auc_sp_mean for c in cats],0.36,color=B,label="Spatial")
ax3.bar(x+0.18,[rc.loc[c].auc_fr_gl for c in cats],0.36,color=G,label="Frontal")
ax3.set_ylim(0.7,0.9); ax3.set_xticks(x,lab,fontsize=7); ax3.set_ylabel("AUC",fontsize=8)
ax3.legend(frameon=False,fontsize=8,loc="upper right",ncol=2); ax3.set_title("Discrimination from\nnormal ECGs",fontsize=9)
for a in (ax1,ax2,ax3):
    a.spines[["top","right"]].set_visible(False)
ax1.set_ylabel("%",fontsize=8); ax2.set_ylabel("Missed, %",fontsize=8)
fig.savefig("Graphical_abstract.pdf"); fig.savefig("ga.png",dpi=300)
from PIL import Image
im=Image.open("ga.png").convert("RGB"); print(im.size); im.save("Graphical_abstract.tiff",compression="tiff_lzw",dpi=(300,300))
