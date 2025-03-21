import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_car_park(rows=2, cols=9, bay_width=2.6, bay_length=5.0):
    fig, ax = plt.subplots(figsize=(cols * bay_width / 1.5, rows * bay_length / 3))
    ax.set_xlim(0, cols * bay_width)
    ax.set_ylim(0, rows * bay_length)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    
    for row in range(rows):
        for col in range(cols):
            x = col * bay_width
            y = (rows - row - 1) * bay_length  # Flip y to match expected layout
            rect = patches.Rectangle((x, y), bay_width, bay_length, linewidth=1, edgecolor='black', facecolor='lightgray')
            ax.add_patch(rect)
            ax.text(x + bay_width / 2, y + bay_length / 2, f"{row+1},{col+1}", 
                    ha='center', va='center', fontsize=12, color='black')
    
    plt.show()


if __name__ == "__main__":
    draw_car_park()
