# %%
from substack_api import Newsletter

# %%

newsletter = Newsletter("https://nosetgauge.com")
recent_posts = newsletter.get_posts()

# %%
