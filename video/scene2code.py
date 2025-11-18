import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.figure(figsize=(12,6))
ax = plt.gca()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Panels
ax.add_patch(patches.Rectangle((0,0), 50, 100, facecolor='#FEE2E2', edgecolor='none'))
ax.add_patch(patches.Rectangle((50,0), 50, 100, facecolor='#FEF3C7', edgecolor='none'))

# Left (Over-Provision)
ax.text(25, 92, 'OVER-PROVISION', ha='center', va='center', color='#B91C1C', fontsize=18, fontweight='bold')
# Tall allocated bar
ax.add_patch(patches.Rectangle((20,10), 10, 75, facecolor='#DC2626'))
# Ghost used bar
ax.add_patch(patches.Rectangle((35,10), 10, 40, facecolor='#DC2626', alpha=0.25))
ax.text(25, 5, 'ALLOCATED vs USED', ha='center', va='bottom', fontsize=10, color='#7F1D1D')
ax.text(25, 88, '🔥', ha='center', va='center', fontsize=40)

# Right (Under-Provision)
ax.text(75, 92, 'UNDER-PROVISION', ha='center', va='center', color='#B45309', fontsize=18, fontweight='bold')
# Allocated short bar
ax.add_patch(patches.Rectangle((70,10), 10, 35, facecolor='#F59E0B'))
# Needed dashed outline
ax.add_patch(patches.Rectangle((85,10), 10, 65, facecolor='none', edgecolor='#92400E', linewidth=2, linestyle='--'))
ax.text(75, 5, 'ALLOCATED / NEEDED', ha='center', va='bottom', fontsize=10, color='#78350F')
ax.text(75, 88, '⚠️', ha='center', va='center', fontsize=40)
# OOM badge
ax.add_patch(patches.FancyBboxPatch((82,78), 14, 8, boxstyle="round,pad=0.4", facecolor='#DC2626'))
ax.text(89, 82, 'OOM', ha='center', va='center', color='white', fontsize=10, fontweight='bold')

# Footer band
ax.add_patch(patches.Rectangle((0,0), 100, 8, facecolor='#111827'))
ax.text(50, 4, 'WASTE   •   INSTABILITY', ha='center', va='center', color='white', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('scene2_over_under.png', dpi=160)