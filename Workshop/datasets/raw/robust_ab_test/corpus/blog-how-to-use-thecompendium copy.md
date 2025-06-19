How to use thecompendium.cards
An incremental way to share developing nuggets of insight
Alex Komoroske
Alex Komoroske

Follow
7 min read
·
Feb 3, 2020
7


3



This post is the second part of a two-part story behind https://thecompendium.cards. The first post covers the underlying motivation and journey to this new format, and this post covers a more practical guide to how to consume and interact with the webapp and its companion twitter bot (@CardsCompendium).

https://thecompendium.cards is a place for me to share developing, inter-linked nuggets of insight in public in an incremental way — think of it as my Medium drafts folder, organized as virtual, inter-linked index cards of developing ideas. Although the format makes it easy for me to incrementally add to and allows representing complex, non-linear ideas, it can also be a bit overwhelming to consume without a series of tools to help light your way through the labyrinth. This post describes those tools and how to best use them, and will be updated over time as more tools are added or changed.

The collection of cards is only loosely ordered, and by default organized primarily by how “baked” they are. There’s no right order to go through them, and it’s easy to get lost in the branching and ever-evolving structure. Most of the tools are focused on helping you not get lost, especially as the content is updated. Note that as you use the web app the URL will always be updated to link to the card and collection you’re currently viewing (which is convenient for sharing), and the back button will work as you expect it to.

What you’ll see when you visit https://thecompendium.cards
The basic webapp
@CardsCompendium Twitter Bot
The @CardsCompendium is a twitter bot that tweets the titles and links of a few cards a day, prioritizing new and popular cards. It’s the easiest way to stay in the loop without getting overwhelmed by all of the content. Many cards will make sense purely from the tweet text, but for those that don’t the link is a great starting point to dig in deeper. Retweets and favorites of the tweets will also show up as stars on the cards in the webapp (more on stars below).


The twitter bot
The ‘Recent’ tab in the webapp, found in the top header bar, is also a great way to see the cards sorted by when they were last substantively updated or created.

Sign-in optional — but recommended
Most of the tools to consume the compendium (described below) save some state to help you reorient yourself. You can use them right away without signing in, and they’ll persist on the device you’re currently using. However, if you plan to consume the compendium on multiple devices, it’s a good idea to sign in with your Google account (it’s just a few taps). All of the state you’ve built up so far will be synced with your account. (Note that if you’ve already signed in on one device, and create state anonymously on another device and then sign in there, you’ll be asked if you want to throw out your temporary state on that new device and use what’s in your signed in state, or not log in.)

Tracking read state
Each card is by default unread until you mark it as read. This read state helps you keep track of which cards you’ve already seen, including by making the links to those cards be a different color. If you’ve looked at a card for more than three seconds, it’s marked as read automatically. This read state is automatically synced across devices if you sign in. If you want to mark a card as unread, just hit the eye icon again.


The read button under each card tells you if the card is marked as read or not
Whether or not a card is already read will also show up as a badge on card thumbnails.


Cards that are read will have a badge in the upper right hand corner in thumbnails
You can see a collection of all of the cards you haven’t read yet by hitting the eye icon in the header at the top.


Tapping the eye icon in the header will show a collection of only the cards you haven’t yet read
Preview cards quickly
If you hover over any link to a card or card thumbnail, a preview of the card will show up after a couple of seconds.


Hovering over a card will preview it
Manage your queue of cards with the reading list feature
When you’re reading a card, you might want to dig into the other cards it links to — but not want to lose your train of thought from the other cards in the collection you’re currently looking at. The reading list is a feature that allows you to add cards to your queue to read next. Think of it like the pattern of opening a link in a new tab as a queue of things to read next, but optimized for the compendium.


Hit the reading list button below a card to add to your reading list

Cards on your reading list will have a highlighted button

Thumbnails for cards on your reading list will show a badge

Links to cards that are on your reading list will have a double underline
But you don’t have to be viewing a card to add it to your reading list. If you hold Cmd/Ctrl and click a card thumbnail or a link to a card, it will toggle whether it’s on your reading list.

To see all of the cards in your reading list, hit the reading list icon in the header.


Hit the reading list button in the header to see all of the cards in your reading list
When you’re done with a card in your reading list, just hit the reading list button on the card to remove it.

Your reading list will sync across devices if you’re signed in.

Star cards you love
You can star cards you love to make it easy to find your favorites — and also to signal to others which cards are worth checking out. A tally of all of the stars shows on the cards (without revealing who starred them).


Hit the star button below a card to star it

Cards you star will have a badge on its thumbnail

You can see the number of stars that have been given a card even if you don’t star it yourself

Tap the star icon in the header to see a collection of only the cards you’ve starred
Stars also sync across devices when you’re signed in.

Favorites and retweets of tweets about a given card will also increase its tally of stars, helping people find the best cards.


Favorites and retweets on a tweet from the bot account will show up as stars on the card themselves, too.
Engage in discussion on a card
If you’re signed in, you can also start a public comment thread on a particular card or reply to an existing comment thread.


Tap the comment button in the bottom right of the screen to start a thread about the card you’re viewing

Comment threads show up in the comments panel for a card. For your own messages you can delete, edit, or resolve the whole thread. Everyone who’s signed in can link to a specific comment, or reply to add a new message to the thread.
Please do comment to provide feedback about any given card!

Find
If you’re looking for a particular card, you can use the find tool. You can press Ctrl/Cmd-F, or tap the search icon below any card.


Tap the search icon to open the find dialog

As you type in the find dialog, cards that contain text that matches are shown.
As with any card thumbnail, you can Ctrl/Cmd-Click to add it to your reading list, or hover over it to preview it.

Read on mobile
If you read the compendium on a device with a small screen, you’ll get a view that emphasizes just the card itself. You can mark cards as read, star them, and add to your reading list, and if you’re signed in they’ll sync to other devices. Currently you can’t view the entire collection of cards at once or navigate to a different collection, but that will be added in the future.


The webapp on a mobile device
View tag collections
Many cards are tagged with subjects they’re about. Tags show up beneath the card they apply to. If you tap a tag, it will jump to viewing the card you’re looking at, in context of the collection of other cards with that tag.


Tap a tag to view the collection of all cards with that tag
Be a part of the process
That’s a brief tour of the features of the app. It’s still a work in progress and a number of things are broken or suboptimal. If you have thoughts or suggestions, or want to stand up your own instance, check out the source code, and feel free to create issues or comment on existing ones.

As I continue adding new features, I’ll update this post with more information on how to use them.

