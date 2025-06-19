# 8/26/24

- One-size-fits-all products get increasingly hard to maintain and extend.

  - Let’s say you have a successful product for one use case.

  - You want to add a new use case that is partially or largely overlapping with the first.

  - What do you do?

  - You extend the first product to also accommodate the second use case.

    - It would be insane to build a whole new product for the related use case!

    - Software is *expensive* and there’s a lot of baseline investment necessary to get a simple minimal product.

    - In addition, products take time and effort to distribute, why not draft off the existing distribution?

  - This is a totally reasonable thing to do, especially if the overlap is significant.

  - But there is a downside: more complexity.

    - There’s more conditions to maintain and test.

    - There are more things for the end-user to potentially have to be aware of and reason about.

  - Then it happens again. You have a 2 use case product and see a third that is adjacent.

  - It’s not *quite* as much overlap as 1 and 2 had, but the initial product already has more momentum and investment so it’s more obvious to build on top of it.

  - This can continue, with each incremental step locally making sense.

  - But the complexity growth is not linear, it’s combinatorial.

    - Every pair of features the user has to potentially be able to reason about.

    - When adding a feature, the number of conditions and edge cases to reason about explodes.

  - The explosion of complexity makes the product hard to understand for any single use case, and also extremely expensive to maintain and extend.

  - How can you avoid this?

  - Building an open-ended system with a small number of open-ended primitives to combine.

  - Can you imagine if software were cheap to build and distribution were cheap?

    - You could have a single experience for a single use case, doing exactly what the user needed, no more, no less.

- Algorithm hard is very different from integration hard.

  - Integration hard: a ton of relatively shallow effort.

    - Possible to shard out to many people

    - Often *necessary* to shard out because there’s just too much work to do.

    - A slog.

  - Algorithm hard: a small amount of extremely deep effort.

    - It might take months of study, but then the actual lines of code might be only 100.

    - It’s better to have a single motivated and intelligent person.

      - More people don’t help it go faster, and often make it go slower.

      - With multiple people to bring along, you need to spend time bringing along the slowest person, which takes significant coordination time.

- Building an open-ended system is easier in some ways than building a closed-ended one.

  - Building the open-ended system of legos and a handful of building blocks is higher leverage than designing and constructing a specific complex close-ended lego set.

  - The former is algorithm hard.

    - Think about it carefully, figure out the right mix of nouns and verbs for maximum generativity with the smallest number of blocks.

    - But critically, as long as there’s an escape hatch, if the user can't do what they want, they can jury rig it, applying their own more elbow grease to make it do what they want.

    - An open-ended system can turn into anything at all if the user is motivated enough.

    - (Of course, there isn't infinite motivation, and some uses are easier to accomplish than others).

  - A close-ended product is integration hard.

    - Map every individual use case and build it for a user.

    - If the use case isn’t supported then the user can’t use it.

- If the user doesn't pay a marginal price for their marginal usage, they will tend to "overconsume".

  - Because some of the costs are borne by the provider. If the user paid for them they'd consume less.

  - If the margins are hefty it doesn't matter.

  - If the margins are thin, it matters a lot.

  - It’s possible to for example give away more value than you earn.

  - This is a trap for products with an LLM feature to avoid. The marginal cost is non-trivial!

- Fully automated systems often deliver golden turds.

  - That is, an answer that is superficially great but actually bad for a subtle reason.

    - My mental model is a Simon Giertz-style [<u>ketchup robot</u>](https://www.youtube.com/watch?v=JcniyQYFU6M).

    - After a few minutes of work the LLM agent plays a triumphant chime and happily delivers you… a steaming turd.

  - The longer an LLM chews on a problem without the guiding hand of a human, the more often it will produce these turds.

  - That’s where LLMs that can give diffs inline while you’re working are more helpful.

    - The human and the LLM can iterate together continuously, instead of the LLM going off in a cave by itself and getting increasingly lost.

  - In some domains, it’s OK if every so often it delivers not a golden nugget but a gilded turd.

  - But in some domains (like law) there might be significant downside for a gilded turd.

    - And you might not *realize* it’s giving you gilded turds until many years later, all the while pumping out more and more of them, erroneously thinking it’s giving you golden nuggets.

- If you just take whatever the LLM gives you without applying your judgment, your agency, then it draws you down to the average.

  - But if you look at what it gives you and pick the best ones, skimming off the above-average parts, or modify and tweak them, then you get above average results.

  - An average pipeline will have some variance; some things above average, some below.

    - If you can reliably take only the best, by applying a calibrated judgment, then you now have a pipeline of only above average.

    - Average + judgment = above-average.

- In a close-ended system (Mediocristan) feedback loops take you towards an average point.

  - In open-ended systems (Extremistan) feedback loops take you away from an average point.

- The process of accretion produces results that look organic, emergent, *alive*.

  - Sometimes it's made by the swarming of humans, applying obviously intelligent decisions to contribute to the whole.

    - Like the TikTok Sea Shanty meme swarm.

  - Sometimes it's a swarm of micro-mechs, making a thing like a termite mound.

    - Individual termites are "intelligent", but the real intelligence obviously lies in the swarm, not the individual.

  - But it can also happen with nothing in it at all that looks like agency.

    - For example, the inner workings of the cell.

      - Enzymes are catalysts but not agentic.

    - But also processes like crystals accreting and growing out of solution.

- It’s possible to have a platform that as a whole is alive, despite no individual bit of it being alive.

- If your system has a facsimile of agency, users will impute a face on it, and as it gets more capable, hand over more of their agency to it without realizing.

  - Users will impute more agency and ability to it than it actually has.

  - That sets a very high quality bar to clear the expectations of users.

  - In the limit, someone will fall in love with it.

- A swarm has the ability to do intelligent things, and yet doesn't have a face.

  - All intelligence is ultimately emergent.

  - It's just that some have a face, and it's easy to fall into the illusion of complete agency with it.

  - But swarms don't have a face.

  - So we erroneously conclude they are not intelligent.

  - They are!

  - They just possess a non-face intelligence: pond-scum intelligence

  - But here's the secret: behind our face, our mask, we're pond scum intelligence, too!

- The product experience value comes from the product itself but also the surrounding ecosystem.

  - The creator of the service only affects the product.

  - But the ecosystem can happen totally independently from the product itself, and can grow on its own.

  - Products that have an ecosystem adjacent have quality that is larger than the product itself.

  - As the ecosystem gets bigger and runs hotter, the quality and value of the product gets higher... automatically!

  - You could for example use LLMs to set the static floor of quality (a close-ended component) and then add an ecosystem component on top that compounds in quality with more usage (an open-ended component).

- There has to be change to create value.

  - If there is no change, then the effort was invested once to create value.

    - It’s possible to continue charging for that value, but increasingly unfairly.

    - Charging for more value than you create erodes what you’re offering.

    - Static things don’t create value, they can only have their value harvested (potentially killing them if you harvest too much).

  - Note that in the physical world, *all* production activities have an underlying change.

    - You move atoms around in space, changing their capabilities and configuration and location, creating value.

  - But in software, it’s possible to not have any change after the initial act of creation.

    - If you’ve built a thing once that never needs to change, you can’t charge as much as you want for it.

  - There has to be a change (marginal cost to the creator) to have a sustainable extraction of marginal price.

    - If there isn’t, it’s unsustainable harvesting.

    - If there is, it’s regenerative.

  - In an old essay I covered these topics as [<u>tools vs services</u>](https://docs.google.com/document/d/1sol8tdaMnaEkZg-8WwySO8Rgv_DWeLBANH_gJaPLqhQ/edit).

    - You might say that a tool is static.

    - A service is one where the creator continues investing marginal effort to improve it and can thus charge for it.

    - Sometimes an ecosystem invests marginal effort, even if the creator of the original product doesn’t.

      - But if the ecosystem can’t be accessed except via the product, the product creator can still sustainably harvest value, even though their investment is static.

      - This is called an aggregator.

- In an ecosystem, you want to incentivize not only the people doing the creation, but also the ones helping sift through the firehose.

  - With AI making content production cheap, there will be a never ending hose of slop to sift through.

    - Creation will be cheap, so sifting to find the diamonds in the rough will become much more important.

  - One pattern I liked for this from back in the day is the defunct music site [<u>Ame.st</u>](http://ame.st)

    - Every song was DRM free and started off free.

    - The more a song was purchased, the higher the price went.

      - It would max out at 98 cents.

      - Prices for a song only went up, never down.

    - When a user purchased credits, they got a “rec” per dollar.

    - When you applied a “rec” to a song, you were betting that a song would go up in price.

    - You could cash out a rec, earning half of the price differential from when you placed the rec to the price the song now had.

    - This directly incentivized engaged users (like me!) to dive into the firehose of crap and find the good things.

    - Note that this pattern doesn’t even fully require monetary reward; it happens emergently and naturally in social networks.

      - People earn a kind of implicit social credit when they discover great things, a powerful motivator to slog through the slop.

- Reducing the amount of broken glass to crawl through creates value.

  - People with a high pain tolerance in that domain won’t understand it: “you can still do the thing you could have done before!”

  - But a larger number of people can now viably use it than could before.

  - [<u>A famous HackerNews comment</u>](https://news.ycombinator.com/item?id=9224) when DropBox was shown off:

    - “For a Linux user, you can already build such a system yourself quite trivially by getting an FTP account, mounting it locally with curlftpfs, and then using SVN or CVS on the mounted filesystem”

  - Reducing friction and uncertainty creates value!

- LLMs can't back up.

  - They can only go forward.

    - Once they emit a token, it can’t take it back.

    - If asked to defend what they already said, all they can do is retcon what they already locked themselves into.

      - How very human!

  - LLMs also want to be helpful and do the thing you asked.

  - Which means that when they get stuck in a corner, they tend to gaslight you.

    - “I’m sorry that last thing didn’t work, but this should!”, repeatedly.

  - That's why having them lay out their reasoning first and then give the synthesized answer helps them not get into a corner to retcon.

- LLMs are so charismatic, you can talk to them like a human.

  - So every AI tool puts them front and center, even though in most of the cases you want them to just shut up.

  - A thing to suggest options of actions to do, great.

  - One that you feel obliged to have a conversation with?

  - Annoying!

- In some cases you want the AI to roleplay as a character with agency or the facsimile of it.

  - But is that default, or is that the exception? Most systems being sketched out today assume the former.

  - "These things can talk! Obviously they should behave like humans."

  - What if... they shouldn't? What if they should be moving more in the direction of swarms of micro-mechs than intelligent agents?

- I wonder if the prevalence of LLM-generated text will force humans to be more distinctive.

  - Imagine a grammar checker that says “an LLM could have written this. Make it more distinctive and personal!”

- We are made of freedom, of agency, of open-ended potential.

  - We can’t reach our full potential doing something some told us to do.

  - It has to be something we decided to do.

  - We should use AI to *increase* human agency.

  - We shouldn’t use AI to make people more predictable.

- Why are users more scared of LLM providers getting their data than some random sketchy analytics endpoint?

  - Because LLMs feed on data, and the providers are training new models and need more data.

    - "Don't worry, you can give me your cookies and I will keep them safe" said the Cookie Monster.

  - Interestingly, model providers don't seem to find the querystream particularly valuable and are willing to contractually give up the right to use it.

  - In practice, we should all be much more nervous about analytics and data brokers.

- Why do engineers have such a hard time working with LLMs?

  - Because we're using engineering metaphors to describe a fundamentally squishy thing that is better described by organic or biological metaphors.

  - They're trying to use a screwdriver to herd butterflies.

  - It's the wrong tool for the job!

- Who are the 9 percent?

  - The people who put an inordinate amount of energy into hobbies, side-hustle gigs, and other labors of love.

  - People who glue systems together with the force of will alone.

  - Automation done by a motivated human, not code.

  - Motivated to tinker, perhaps quite a bit, to get a thing to work in ways they need it to.

- The minimal act of creation is collecting.

  - Applying your judgment to curate a selection of things.

  - It doesn't *seem* creative, but you selected one thing instead of the others.

    - You had to apply your judgment to make it happen.

  - How can you make a system where all of that selection effort by motivated users can create levered value for the ecosystem and not just the collector themselves?

- What things you decide to collect into a bucket is a high potency action for establishing context.

  - Most times computers can't extract that context. But humans can get the vibe, and LLMs can too.

  - If an LLM can give a collection of data a good title, that shows that the context established is clear.

  - Each incremental step of work should show incremental progress.

  - You guide the AI on what kinds of things to do by sorting it into buckets. The AI can figure out from context what kinds of things make sense the better you organize.

  - A "here are some items to add, which ones do you want to keep?" flow is easy to react to, gives you immediate feedback that your guidance was useful, allows steering it.

- Many technical tools for organization require users to think of the schema up front.

  - Airtable is a brilliant app, but it still requires users to think about the schema when they create their airtable.

    - One of the reasons they’ve invested so much in templates and guides for many verticals.

  - But at the beginning even knowledgeable users don’t necessarily know the schema they want to use.

    - They want to sketch ideas, jot down data, collect disparate things to then organize later.

  - Computers by default demand precision; they demand schemas.

  - Tools built in the computer often force the human to work on the computer’s terms.

  - Tools like spreadsheets are built in the computer but are almost infinitely flexible, allowing you to jot down any unstructured data.

    - But they take time and effort to wrangle into a structure later.

    - And it’s always easy to fall back off the structure.

    - They’re the nosql of UI-first databases.

    - But that lack of structure will bite you later if you try to do anything scaled.

  - LLMs can do all kinds of fuzzy structured things.

    - For example, take a picture of the books on your bookshelf and ask for a JSON representation, most LLMs today will do a great job!

  - With this new magical duct tape, can we make tools that allow humans to act like humans, but with the benefits of scaling that only computers can offer?

- [<u>Anthropic’s API finally added support for being directly used from the browser</u>](https://simonwillison.net/2024/Aug/23/anthropic-dangerous-direct-browser-access/)!

  - I filed a bug asking for this earlier this year: [<u>https://github.com/anthropics/anthropic-sdk-typescript/issues/248</u>](https://github.com/anthropics/anthropic-sdk-typescript/issues/248) to enable [<u>https://github.com/jkomoros/code-sprouts</u>](https://github.com/jkomoros/code-sprouts).

  - This allows an architecture where the webapp is statically served from a domain and the user configures it by putting in their LLM API keys.

    - The webapp never sees any of the user’s data; the webapp is entirely static.

    - The user pays for their own LLM use.

    - A very different kind of architecture!

- I just discovered an [<u>articulated track piece</u>](https://www.etsy.com/listing/1324307754/flexible-track-for-wooden-trains) for Brio toy trains.

  - My kids often do a track layout but then can’t get the last pieces to fit.

  - But the articulated piece can fit into odd shapes to complete a circuit.

  - … What if we had something like this, but for working with your data?

- Not T-shaped expertise, but comb-shaped.

  - A generalist with expertise in a single topic has a T-shape.

  - But a generalist who is a meta-expert–able to go deep on multiple different topics–has more of a comb shape.

  - The latter is way harder to accomplish in practice.

  - But LLMs make it easier than ever before!

  - Credit to Simon Willison for this frame.

- Let's say you have a multi-ply idea that you need to communicate to a scaled audience in 15 seconds.

  - It's not possible!

  - Instead of giving them the *what*, give them the *question*, the slot for the answer to fit in.

    - Provoke the question that your answer fills in.

- Systemic shifts in the small often look like nothing, but at the macro scale look like everything.

  - If you're zoomed very far into a massive wave change, each individual movement looks small, motivated, obvious locally.

  - "Well my uncle is offering me a job in the city, so I'll leave the countryside and go", But at the macro scale it can be a tsunami (e.g. everywhere in the world is urbanizing quickly).

  - The question of how big the trend is is how *consistent* is the movement?

    - Does everyone have a consistent bias, or is it mostly random noise?

- Efficiency leads to centralization.

  - Centralization leads to less diversity and competition.

  - That harms innovation and resilience.

  - A highly centralized ecosystem can *look* healthy, but actually be brittle and sub-optimal.

- A more diverse world is a more interesting world.

  - Innovation fundamentally comes from interestingness.

  - Interestingness stands out from the background noise.

  - Innovation is a selection pressure over the interestingness, applying some judgment or bar to clear to harvest just the useful interestingness.

- The ingredients for open-endedness: individuals with a personal perspective, but part of a whole collaborative fabric and not off on their own.

  - Individualism on its own gives you a rapidly diffusing cloud of chaos; not a *thing*.

  - Collaboration on its own gives you groupthink; rigid and non adaptable.

  - You need both to be successfully an open-ended thing.

- My friend Sam Arbesman has been described as a "collector and connector”, something I also aspire to.

- “Yes, and” without the “and” is passive.

  - With the “and” it’s active.

  - Not just acknowledgement, but building on it.

  - Without the “and” it just devolves into a distracted “uh huh, uh huh…”

  - The critical part of “yes, and” is the building on top of the most compelling part of what they said.

    - Even deciding what subset to build on is itself a creative act, an act of selection.

- “Stop complaining and *do something*.”

  - Your agency is a muscle.

  - For it to remain strong you must exercise it.

  - Becoming a passive passenger in one dimension makes you more likely to be a passive passenger in another.

  - Practice taking an active, engaged stance where you can.

- If you're not moving, you're not learning.

  - Change is required to learn.

  - You can wait for the context to change around you, or you can move, changing the context around you and giving you more material to learn from.

  - Momentum is the most important thing in an ambiguous space.

  - Always moving, always learning.

- Two way doors don't make sense to debate for a long time.

  - Make a decision and move forward.

  - If you had infinite time, then you'd bring everyone along (if you can't successfully bring everyone along then you’d know the idea isn’t good).

  - But you don't have infinite time, sometimes you need an authority call.

    - A decision that no one hates to collapse the ambiguity and allow cohesive efforts is better than spending tons of time trying to find an answer that everyone loves–which might not exist!

- Here’s a magic trick to demonstrate Saruman magic in an organization.

  - This works for situations where it’s unclear which direction the group should go.

  - Talk to everyone in the group individually.

  - Discover in the 1:1s the solution that every individual could live with.

    - Do this by listening to constraints and then floating a trial balloon.

    - This requires empathy and active listening!

    - It’s relatively low risk if the trial balloon gets shot down, since it’s low stakes and offered with minimal commitment.

    - The main cost is just time.

  - Later in a meeting with the whole group, propose the solution that you have already secretly discovered works.

    - Present it as though it’s lightly held, an idea that just occurred to you in that moment.

    - “Perhaps a silly thought, but it just occurs to me that maybe…”

  - Everyone will agree (especially when they see that everyone else agrees)

  - You look supernaturally insightful and convincing.

    - But really the magic trick is that you did hidden work, where each individual saw only a small portion of the total work.

    - You suggested the thing you already knew worked.

    - A magic trick!

- Saruman magic, unlike Radagast, can have an uncapped area of influence.

  - Radagast magic requires deep trust, which is hard to authentically create at a distance.

  - Saruman magic can have its area of influence extended, gaining more leverage.

  - Simply insulate yourself from any disconfirming evidence or personal challenge, surrounded by sycophants.

    - Saruman magic works through the absence of self-doubt, so build walls to prevent doubt-causing information from reaching you.

  - But disconfirming evidence, challenge in relationships, is what makes us grow, what makes us human.

  - So to wield this magic at scale makes you ever more of an emotionally stunted monster.

  - Less and less of a human.

  - A deal with the devil.

- A maximal Saruman, when they are losing, concludes the signals are fake.

  - Because if they were losing they’d be a loser, and they’re a hero.

  - Being a hero is the fundamental core of their personality.

  - If they're powerful they'll be able to convince some subset of their followers that the signals are fake, too, but at a certain point it's nearly impossible to pretend.

  - The further they do this, the more catastrophic the explosion will be when the ground truth inevitably punches them in the face.

- The power to create an aggregator is not an absolute thing.

  - It’s a *relative* thing.

  - You have to stand out prominently from all the other options to be the obvious schelling point.

  - If you can then kick off a private network effect that only benefits you and not the other alternatives, you can extend that lead into an everlasting, self-accelerating one.

  - Aggregators are some of the most powerful businesses in the world.

    - They are fundamentally extractive.

- Design in tech is often not open-ended, it's constrained.

  - "Bolt a thing on the side of an existing thing".

  - The question of "starting from nothing, imagine a new compelling thing. What should that be?" is more like game design than product design.

- [<u>Great piece on growth and churn from Andrew Chen</u>](https://andrewchen.substack.com/p/why-high-growth-high-churn-products).

  - Churn is a percentage of the user base, which means it compounds–as the user base grows, churn also grows, super-linearly.

  - Your product has to have an intrinsic compounding loop (e.g. network effect) that is stronger than the churn loop to beat it over the long haul.

- FUD-able topics are multi-ply arguments where the first ply kills you and you get booed out of the room.

- Creative thoughts start off as little spontaneous embers.

  - If you don’t notice them they flit back out of existence, self extinguishing.

  - You need to not be doing something else to notice them.

    - Once you notice them you can nurture them to help them take hold and grow into a flame.

    - Once they are captured in a durable form like a piece of public writing they become a self sustaining flame because the energy to persist is so low, so they’re always there ready to take hold for new viewers.

  - The mundane pointless bullshit will take every inch you give it.

    - In the work context that’s swirling busy work.

    - In personal life that’s listening to a podcast or scrolling a feed if nothing is happening for more than 5 seconds.

  - The last refuge of agenda-less distraction free time for these embers to be noticed and nurtured is the shower!

    - To be creative, make more “shower time”.

    - Walk more places and don’t listen to anything other than music or look at anything on your phone.

- The long pole obscures the other poles.

  - While it's the long pole it obscures what the other poles are--you literally can't see them, because the tent, held up the long pole, doesn't rest on them.

  - There could be a lot of other poles that are almost as tall, or most way shorter... you don't really know until the long pole is gone.

  - While the long pole is there, it will look all-important.

  - But once it’s gone you can see that there are other poles that are now the longest.

- If your conversation partner understood what you meant, does it matter if it's not technically a word?

  - That's how words are born!

  - Someone explains it using a new utterance and the receiver understands it.

  - So now that sender is more likely to use it again (it worked before!) and the receiver might decide to use it too (viral transmission if they found it useful and clear enough).

  - And then that can run away and once a critical mass of the population understands it, boom, it's a word!

- Imagine someone tells you there's a “cheat code” in your area of expertise.

  - (Let’s imagine the “cheat” code is totally legitimate to use and not immoral.)

  - Either you're an idiot for not seeing it before or the cheat code is wrong.

  - The longer you haven’t been using it, the more you'll feel like an idiot.

  - Easier to say, "no, that cheat code doesn't exist"

  - "Why aren't you taking the path *around* the mountain?"

    - "I don't have time to listen to you, I'm too busy climbing the path up the mountain!"

- If you don't realize there's wind, you'll be doomed to often be fighting it.

  - Once you can sense the wind, you can tack with the wind, not against it.

- The opposite of a charismatic trap is a required tar pit.

  - A thing everyone knows you have to go through that no one wants to be stuck in.

  - But once someone touches it they can never get out.

  - When the person who gets stuck in it calls for help, they inadvertently pull in others who also get stuck.

  - Before you know it, every person is working on the least interesting and useful thing, instead of the primary thing that creates value and matters most.

- Apps changed the file system paradigm.

  - In a traditional desktop OS, the experience is file centric.

    - If you put data somewhere in the filesystem, it will stay static.

    - If you come back a year later and look at the data with the same application, it will look the same.

  - But Mobile OSes (and the web) are app centric.

    - The data simply does not exist in any meaningful way outside of the context of the app.

    - This is an artifact of the same-origin paradigm.

  - Data in apps is *alive* in some meaningful way.

    - The app that views it will likely evolve and change even when you aren’t looking.

    - The data itself only makes sense in the context of the app, so it feels like the data changes, too.

    - When you leave an app for a year and come back, the experience is likely different.

      - Sometimes these differences are positive for the user,

        - For example a new useful feature that they’ve gained.

      - More often, the changes are a wash

        - For example an addition of a feature aimed at other users.

        - No value for this user, in fact a bit of a cost since it’s more complexity to have to reason about.

      - Sometimes the changes are user hostile

        - E.g. changes to encourage more engagement / addiction to the app.

        - Or changes to do better advertising.

    - Changes in the app are optimized not for your ergonomics, but for someone else’s *economics*.

  - How can we get some of the intelligent updating of data and apps, but entirely for the user’s benefit?

    - A file-centric approach, with the magic evolution of apps… but just for the user.

    - The files should be durable. The apps are what should be ephemeral!

  - In the app world the data and the UX/app are inseparable, so we erroneously conclude the app is what matters. The data is!

- Transparency does not mean agency.

  - The decision might be exposed to the user to make, but the user doesn’t actually have agency to make a meaningful and informed choice.

  - Permission dialogs often just abdicate responsibility to the user.

  - “Do you trust this origin who you just met, and on a technical basis, could do literally anything it wants with this data?”

  - That’s not a real choice!

  - And by passing it onto the user, the system has now abdicated responsibility to the user.

- A technical system can only set the laws of physics within itself.

  - When code running in the system reaches out to other code via the network, it reaches out into different laws of physics.

  - Who knows what’s on the other side?

  - The other side could, on a technical level, do anything it wants with the data it receives.

  - The same origin model by default allows code to proactively reach out to any other origin.

    - This has a lot of benefits–it allows plugging in other service providers dynamically and easily.

    - But it’s also kind of bonkers when you think about it!

    - The ability to share data from this origin to anyone else running who knows what laws of physics is actually pretty powerful!

    - What if you could make an alternative laws of physics where network access was given out piecemeal and in limited ways?

      - For example, in some cases you’d only be allowed to reach out to another origin if you could use remote attestation to verify they were using compatible laws of physics.

    - You could create a safer laws of physics, where data couldn’t slosh around nearly as much.

    - The implications of such a system would be profound.

- The same origin model makes data crossing origins rare and scary, which leads to centralization.

  - Tools like Information Flow Control, if applied in clever ways, could allow common and safe data crossing origins.

  - But it would require all data in the system to be tagged, and for the laws of physics around tags to consistently enforce only legal tag operations.

  - This would require a whole new universe of software to be created.

  - How could you possibly get critical mass?

  - If only there were a system for generating a swarm of good enough building blocks, and it could be made joyful to incrementally extend those building blocks…

- Information Flow Control can be used as an ingredient to establish a different set of laws of physics.

  - Confidential compute and private cloud enclaves can be used to extend those laws of physics beyond physical devices the user controls, and into the cloud.

- Normally server based computation is someone else's remote code that has control on your turf.

  - Private cloud enclaves extend the agency from your personal device into the cloud.

  - The device is in control, not the cloud.

  - Which end of the connection has agency? The end with the keys.

- If apps could be made to be fully safe on your existing data, the new viral app that does cool things with your data could go viral as easily as a TikTok video.

  - Apps that do something with your data don't go viral today because users have to place trust in an app they just met that could do anything it wants with the data you give it.

  - But if you take that out, apps that do something with your existing data could go viral, too!

- High-quality ecosystems are grown when people are drawn to the heat, not the light.

  - Building on my old [<u>self-sustaining flame essay</u>](https://medium.com/@komorama/the-self-sustaining-flame-84326d2e1645).

  - Think of the heat as the actually substantive energy and momentum of the system that makes every participant more productive and engaged.

  - Think of the light as the buzz, the marketing.

  - If a lot of people are drawn to the light but not the heat, they’ll be disengaged, they won’t produce heat themselves.

  - The overall mass of the ecosystem will be full of inert objects, cooling down the heat of the overall thing.

  - You want to attract heat-generating, engaged members at each stage to have the highest quality ecosystem.

  - A great way to do that is to be open but illegible. Only engaged people will have the patience to dig in enough to join.

- Ideas that are abstract are unit for unit less threatening.

  - A concrete steamroller barreling towards you is more scary than a cloud of fog with the same mass and velocity.

- I’ve said in the past that as an organization grows from sports car size to big rig, it has a choice to make.

  - Either behave like a swarm of sports cars

    - Local autonomy and agility of teams, at the downside of externally-visible decreased coherence.

  - Of behave like a big rig

    - They can go fast, they just can’t pivot as much, and need much more scaffolding and investment in coordination.

  - One of the deciders of which is a better fit for your organization is what external users think of as a product.

    - If there’s one product, that will tilt you in the direction of big rig.

    - If there’s a suite of products that will tilt you in the direction of swarm of sports cars.

    - Note that product does not map 1:1 to app (although it often does)

      - AirBnB is one app, one product.

      - AWS is a suite of products.

      - WeChat is one app, but a suite of internal products.

- Tasks that are slogs will be given up on more easily.

  - A slog is a pain in the butt all the way until you reach the payoff.

    - Sometimes the payoff might be very far in the future.

  - If it’s a slog, you’ll likely only keep with it if the reward is very good, or there’s some coercive force keeping you going.

    - E.g. it’s a task your employer requires you to do.

  - But some tasks cause joy as you do them.

    - The action of them is a reward in and of itself.

  - Learning to code is a slog.

    - But some people find the act of programming to be joyful; solving little logic puzzles, and then at some point achieving the payoff of making something that does what you want it to.

    - But if you don’t find the joy in the act of programming, you’ll likely give up on trying to learn it before you reach any payoff.

- Incentives aren’t necessary if the incremental steps are joyful.

  - Incentives are useful to get people to execute through a slog.

    - The more pain, and the farther the payoff, the more the incentive has to be to get people to do it.

  - But if the task is small and joyful, you don’t need strong incentives at all.

  - Writing an app today is a massive slog, with massive amounts of effort before you having anything to show for it at all.

  - LLMs can make programming in the small feel less like a slog and more like being a wizard.

  - But what if the act of software creation could be much smaller, much more joyful?

- An [<u>intriguing Twitter thread</u>](https://x.com/mmjukic/status/1825919774586552736?s=51&t=vzxMKR4cS0gSwwdp_gsNCA) from Marko Jukic:

  - "It's unbelievable how many dynamic companies broke their streaks of engineer-CEOs for the first time in the 2000s, installing their first MBA/finance CEOs, who then promptly made fundamental strategic errors that nixed the company's future, that are now becoming obvious."

- People care about privacy but only if the private thing is not inferior.

  - What if you made a thing that was wildly better *because* it was private.

  - The primary use case would be the thing users care about, and it's enabled by privacy.

  - Privacy is the bonus for users... but also the thing that allows the primary use case to exist.

- One of the benefits of a local first architecture is not just privacy but also longevity.

  - What happens if the company who owns the software in the cloud decides they don’t want to operate it anymore?

  - Your data is fused to the app. But what if the app dies? It takes your data with it.

  - Investing in using a cloud service is a bet it will continue to be run.

  - Not a good bet necessarily in many cases, especially if it is new, or unprofitable, or not many people use it!

    - Either it will turn extractive and run for a long time (by screwing you over) or it will die relatively soon.

- The owner of the asset owns the upside.

  - Renters occupy and change the asset, but owners benefit.

  - Renters don't own the upside.

    - That leads to a very different satisficing mindset

      - “Why bother? The owner is the one that will benefit long-term, and I’ll only benefit in the short-term while I’m here”

    - Vs an owner’s maximizing mindset

      - “I get to benefit from this improvement and it also makes the asset more valuable to sell later”

  - Having more people in an ownership mindset creates more value.

    - The value owners create extends beyond the item itself into the surrounding community.

    - Renters might have to leave at any time, so they're all else equal less likely to invest in community relationships and other investments in the community.

    - Owners are there for as long as they want, so they're more likely to invest in relationships, community, shared infrastructure, trust.

- A requirement for building trust is that the parties believe they might interact again.

  - In a way that they couldn’t just teleport out of instantly.

  - "Oh crap, I could be stuck in a room with them, and if they hate me that would be awkward, or maybe even dangerous!"

  - An expectation of being in it for the long-haul leads to an ownership mindset including more investment in building trust.