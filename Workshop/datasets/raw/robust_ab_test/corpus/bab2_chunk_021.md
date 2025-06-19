# 1/27/25

- Talking to an LLM is like talking to a choir.

  - Or maybe a Greek chorus.

  - Sounds like one voice but is actually a multitude singing coherently.

  - Not a single perspective, the collective, average hive mind of humanity.

  - Talking to everyone and no one.

  - Haunting, larger than life.

- Without humans in the loop LLMs just produce slop.

  - Humans are the curatorial energy that help find the greatness amid the cacophony and extract it.

  - The right answer is not “how to have only LLMs in the loop with good output” but “how to efficiently interweave humans and LLMs in high-leverage co-creative dances”

- People are using AI to solve yesterday's problems, not tomorrow's problems.

  - Doing yesterday's thing 20% better will not win in a disruptive context.

  - In the face of disruptive changes, throw out the "efficiency" playbook.

  - Instead, ask: “What new and transformative things are enabled by this disruptive technology?”

- LLMs are more a sociological phenomena than a technological one.

  - LLMs are best thought of as sociological mirrors of human society, not technological artifacts.

  - Adjacent to Alison Gopnik’s Stone Soup AI frame.

- Remember: we’re at the “filming stage shows” stage with LLMs.

  - In film, it took awhile to discover the power of montage, a particular superpower of the medium.

  - We don’t yet know what the particular superpower of the LLM medium is yet.

  - We’re waiting for the LLM’s montage moment.

  - Once it happens, it will feel totally obvious in retrospect.

  - We’re in the pre-montage phase of LLMs.

- The challenge with LLMs is often giving them the right context.

  - It’s not that they lack a baseline common sense, it’s that they don’t know anything about your particular situation if you don’t tell them.

  - This is an insight from Jesse Andrews in the comments of last weeks’ reflections.

  - Often it’s hard and tiresome to tell them all the background knowledge you need them to know.

  - LLMs are in a dark room; each conversation is a fresh start where you have to start from scratch giving it background knowledge about your particular goal and context.

  - This all implies to me that the LLM “montage” kind of breakthrough will be around wrangling contexts.

- [<u>A simple trick</u>](https://x.com/voooooogel/status/1881966969043464365) to change how LLMs reason: every so often, inject “Wait, but ” into the in-progress reasoning token stream.

  - This forces the LLM to reflect on what it might have missed, giving more resilient output.

  - Another hint that managing contexts will be the “montage moment” for LLMs.

- My friend Anthea’s name for the [<u>new LLM-assisted “book” medium</u>](#pottbo1qsnvc): a liquid textbook.

- I've talked in the past about how chatbot UX feels wrong.

  - It's too limiting, leans too hard into the artifice that the LLM is "just a normal human", when in reality it's a collective hive mind of humanity that has orders of magnitude better recall than any human... and also lacks some aspects of common sense.

  - I see dialogues with LLMs as not a conversation but an act of co-creation, and a chatbot UX seems to undermine that, and foreclose on other interactions.

  - An alternate UX is kind of like a Google Doc, where you could hit Shift-Enter to ask the LLM to autocomplete from your cursor for as long as it wanted to.

  - It is more of a "based on what comes before in this doc, keep generating as many tokens as you want, drawing on your hive mind of humanity."

  - You could still do a normal append-only chat log if you wanted, just each time before you hit Shift-Enter, you'd type: "Computer:".

  - But you don't have to do that if you don't want to, and you can also clean up the stuff earlier in the doc to hone the context to get closer to what you want, to steer it better.

  - Kind of like GitHub Copilot but where instead of tab adding a few characters, it could add whole paragraphs.

  - Because the key command is not simply "Enter" (which would add LLM-generated slop after *every* line you write), the user has to choose to call it into being.

  - This also allows patterns where you start the LLM off going down a particular line of thought, or combine insights from another LLM’s thinking tokens to help direct another LLM’s conclusions.

  - Kind of a simple, obvious interaction pattern, and not that different from chatbot UXes, and also more similar to the *original* LLM completion APIs from a couple of years ago.

  - But it feels like it's truer to the materials and what they're good at than the current chatbox interfaces.

- Alpha-zero showed how much learning could happen if you had a rigorous ground truth system.

  - Games like Go have rigid, clear rules about what moves are legal and what constitutes a win.

    - The ground truth can be applied without a human in the loop, because the rules are black and white and possible to easily model in a computer with full fidelity.

  - That means if you set up a co-evolutionary loop, you can pour extraordinary amounts of compute into it and it will get better and better, with no humans in the loop.

    - A self-catalyzing infinite stream of training data.

  - That hasn’t worked for things like reasoning yet because there’s no ground truth you can efficiently compare against.

  - But now GPT4-class models are commodity and there are a number of open weights versions.

  - Those models can act like the “ground truth” for other models to use to bootstrap off of… it just requires conveniently ignoring the license of the open weights models.

  - Some of the recent breakthroughs likely happened in this way.

    - “We just released an MIT-licensed Llama-derived model.”

    - “Wait, what?”

  - It’s impossible to imagine that this can be stopped, it’s too powerful a technique, and too easy for someone to have a licensing “oopsie”.

    - Once the weights are published, there’s no taking them back.

    - By the time the original publisher of the infringing model is taken down (which might take a long time, especially if they’re international) that derivative model has been picked up by the swarm.

- The more time you're fiddling with GPUs, the less time you're playing with the AI to develop intuition for what it can do.

  - LLMs are primarily a sociological phenomena; understanding them at the object level of running them doesn’t tell you what you can use them for.

- Right brain energy will become more important in a world of LLMs.

  - Left brain energy is structured, convergent.

  - Right brain energy is creative, divergent.

  - LLMs can do convergent, best-practice thinking quite well.

  - LLMs struggle with divergent thinking (without the correct thought partner).

  - If you specialized in convergent thinking, you might find your skills less differentiated in this new era.

- Claude’s love language is React components.

  - Any programming-adjacent question you ask it, it hops immediately to writing a crappy little React component without fully understanding what you want.

  - If you want it to think before it leaps you have to hold its hand.

    - “No code yet! Just think about how we should architect this.”

    - “Let’s take a step back.”

- The trick in this new era will be to figure out how to weave in human taste with leverage.

  - The LLMs shouldn’t decide if something is good, the humans do.

  - If you can create mechanisms to reflect back those human quality decisions so others in the ecosystem benefit, you get the best of both worlds.

  - An emergent, leveraged, humanistic taste, not slop.

- Why do we have the chonky apps of today? Here’s the Coasian theory of the app.

  - The Coasian theory of the firm is that the size of firms is tied to the surface tension of the boundary of the firm.

    - Things that are possible to formulate as an easy-to-administer contract will be done outside the firm.

    - Things that are not possible to formulate in that way will be done internally.

  - Let’s apply this concept to apps and software of today.

  - The default “size” of apps (in terms of number of features, number of screens, number of lines of code) comes down to a few factors:

  - 1\) The cost to write software

    - The more expensive, the more features have to be bundled around the business model to make the fixed cost investment worth it.

  - 2\) The cost to distribute software

    - How high of friction is it to get the bits onto the user’s phone to interact with?

    - Apps are high friction: big chonky installs. It costs multiple dollars per install in marketing.

  - 3\) Coordination costs across apps.

    - This is the core of the Coasian frame.

    - How hard is it for an app to coordinate with neighboring apps?

    - The harder it is to share data, the higher the surface tension, and you’ll get a smaller number of larger apps.

    - The same origin paradigm makes each origin a separate universe, and makes data transfer between apps extremely hard.

    - This leads to chonky apps.

  - AI with the right programming model could change all of this considerably.

    - Leading to a swarm of small, simple apps coordinating emergently.

- Shitty software in the small is now practically free to create.

  - Everyone’s trying to produce large, chonky software of today in it.

    - It’s hard to squeeze enough quality while shoehorn it into the app creation flows programmers use today.

  - But what if we leaned into an architecture that *presumed* shitty software in the small?

  - Where the innovation was not each individual bit of software, but their *combination*?

  - What would have to be true in that architecture?

  - A totally different way to architect software, that is AI-native.

  - Don’t use the disruptive tech of LLMs try to build apps of today 20% faster.

  - Build new kinds of apps.

  - Shitty software in the small allows infinite disposable component.

- Disposable components might lead to a different, more fluid UI paradigm: liquid software.

  - Apps are carefully designed and composed UIs.

    - Every widget is carefully laid out and considered in the overall design.

    - Integrating new features into an existing UI is extremely challenging and requires taste and careful consideration.

  - LLMs are great at shity software in the small, but tastefully integrating novel features into existing complex software is currently a bridge too far.

    - It requires a human in the loop (possibly with an AI exoskeleton).

  - What if the right UI is not a single well-considered UI, but a progression of little individual UIs that all coordinate and work on the same underlying data.

  - If the UI doesn’t do the thing you want, conjure up a small UI panel to do precisely the change you want, and then throw it out when you’re done.

  - Slap a disposable component on top, and throw it out when you’re done. The overall emergent assemblage is not hard, but liquid.

- The last era of software was based around zero marginal costs.

  - In a world of zero marginal cost, there are only three consumer business models.

  - Hardware.

    - Charge a premium on the hardware, and lock people into your ecosystem.

  - Media.

    - Proprietary copyrighted content the user can’t get anywhere else.

  - Ads.

    - This is the default catchment basin nearly all of consumer fell into, an inescapable pull.

    - But to have a sustainable ads ecosystem requires a critical mass.

    - This is a heavily centralizing force, especially with things like Apple’s ATT which traded off a small increase in privacy for massive centralization.

  - The ads business model leads, inevitably, to engagement farming.

    - Giving users not what they want to want, but what they want.

    - The old saw “You are the product” has some truth to it.

    - Incentives between the software and the users are somewhat at odds.

  - LLMs have too high a marginal cost to be supported solely by advertising.

  - Good!

  - That means the next era will be free of that default catchment basin, and hopefully we’ll find some more user-aligned incentives in the LLM-native catchment basin.

- If you want Google to change a feature just for you, you’re one speck out of billions.

  - Of course they won’t.

  - They would only do it if one of the employees thought it was in their business interest, and was motivated enough to coordinate a bunch of other employees.

  - You are one individual molecule in a vast ocean of users.

  - At that scale Google can’t think about what individuals need–there’s far too many.

  - Centralized software requires treating your customers as a vast anonymous *market*, not as individuals.

  - It would be like asking Kraft to make a version of Kraft Dinner with your favorite flavor.

    - Impossible. Ludicrous. Unthinkable.

- Software today is shrink wrapped and pixel perfect because it has to be to justify the distribution cost to dissimilar users.

  - Software is expensive to write, so building it requires identifying an audience (as large as you can) and then building a thing they all would like… which necessitates aiming for the lowest common denominator.

  - Once it becomes mass produced, it loses its soul.

  - It feels less like human-scale creation for individual humans, but machined, efficient creation for whole abstract markets.

- Building technology is primarily an engineering problem.

  - What it's used for is primarily a social problem.

  - You can't tackle social problems with engineering approaches.

  - Tech today, in this late stage of this era, is not about technology but about business.

- I would prefer cozy tech over big tech.

  - The tech isn't the problem, the big is.

  - Big makes it so you can't escape the gravity well.

  - You have to hope the gravity well decides your need is important and aligned with its business interest.

  - If not... you don't really have many alternatives, because they've all been sucked into the gravity well, too.

- Cozy software is humanistic software.

  - Software lost its soul in the mass produced era.

  - Optimized for extraction, mass market, not value creation.

  - Treating users the same, not leaning into what they actually individually want.

  - LLMs provide the opportunity to change this.

  - Humanistic computing. Human scale software.

- Technology on its own is great, an optimistic force for humanity’s thriving.

  - It’s the tech broligarchy part that’s the problem.

  - The tech is fine, it’s the centralized power structures and hyper-financialized approach I don’t like.

  - In the past we had many ecosystems we could pick from some of which were walled gardens.

  - Now we have a handful of options to pick from, all of which are walled gardens.

    - “[<u>Oops, all walled gardens!</u>](https://knowyourmeme.com/memes/oops-all-berries-box-parodies)”

  - This hyper centralization is not an intrinsic property of technology; it just so happens to be where we ended up within this set of physics in the late stage.

  - So let’s change it!

  - Now is the time to reset the power dynamics, given that we have a disruptive technology bursting onto the scene.

  - The birth of a new early stage where we can put the tech to humanistic ends.

- Our personal context was shattered into a million pieces, in a million separate pocket universes.

  - That’s the curse of the same origin paradigm.

  - Each app or website is a totally separate universe.

  - Some universes (the aggregators) have tons of our data, but do very little for us as individuals.

  - We as individuals need to chase and try to wire together over various domains.

  - We need to do the work to get the information into one place.

  - That means that we often just give up; there’s so much value we could create for ourselves from our data if all of the combinatorial power of it was in one place that was optimized for *us,* not some other business.

- The fact that Reddit “owns” the data their communities created is an accident of the same origin model.

  - The entity that wrote the software and pays the server bill owns the rights to the data.

  - … wait, what?

- Sometimes early momentum traps you, a phenomena you could call creating a tech island.

  - An early mover with a clear advantage builds up their proprietary system.

  - Over time, the outside world creates an open alternative.

  - It starts off not as good, and the early mover continues to have an edge.

  - But the early mover is one entity, and the outside world is a swarm.

  - The swarm can innovate faster than the singular entity, so it starts off slower but gains momentum and inevitably beats the early mover.

  - But now the early mover is stuck in a terrible dilemma

    - Option one is to continue with their internal system.

    - Option two is to switch to the external system.

    - Option one seems tempting.

      - It used to give them a true edge, after all… maybe they can recover that edge.

      - But the open swarm will overpower them and leave them further and further behind, making the dilemma even worse.

    - Option two seems terrible.

      - To switch to the external system would give up all of the advantage, and actually put them at a disadvantage since they have to learn how to do things in a totally different way at odds with all of their previous knowhow and expertise.

  - This dilemma gets harder and harder the longer it goes on.

  - Early momentum that then holds you back.

  - Stuck on an alternate branch of history.

  - This dynamic only happens in the early stage of a paradigm, not the late stage.

- A similar trap early movers get stuck in: AOL-ing yourself.

  - A bet of being a proprietary walled garden with an early fast and powerful start, but going up against an open system.

  - Leaning into your proprietary, non-scalable advantage in a way that will increasingly trap yourself.

  - OpenAI has an initial proprietary advantage that it’s focusing on going all in on the consumer aggregator / walled garden approach.

- It’s your agency.

  - Don’t give it away.

  - If you were tricked into giving it away, take it back!

  - Your data should work primarily for *you*.

- LLMs are like clay. Normal programming is like legos.

  - Precise, unyielding, but flexible in combination.

  - LLMs are inherently flexible. For some use cases, too flexible!

- What makes Lego such an amazing, open-ended system?

  - Part of it is the lego dot; the way that all of the pieces fit together.

  - But that’s not as important as it looks; there are a number of different dots that would have worked.

  - What’s more important is the extremely tight levels of precision in the manufacturing process.

  - That’s what allows every lego, even ones made decades ago, to fit perfectly with every other piece.

  - Each piece is inflexible and hard; but in combination the system has extraordinary flexibility.

  - The shape of the dot is arbitrary.

  - What matters is the precision.

- Conversation is not primarily an act of communication but an act of creation.

- You can be creative and introspective alone, it just is way less fun, and way harder.

  - In the right team environment, you excite one another.

  - The "Yes, and" from another gives you another burst of energy and optimism to do the next turn.

- No individual web page could convince you of the value of the web.

- The web was an overwhelming tsunami of information.

  - But Google made it trustworthy, easy to navigate, tuned to what you wanted, more calm.

    - The overwhelming cacophony of the web paired with a trusted personal guide.

    - A powerful complement, a phenomenal business.

  - AI feels like a tsunami coming for us–overwhelming, impossible to avoid.

  - What will be the product that makes AI trustworthy, that makes it work for us?

- To have good, calibrated taste requires you to have dabbled in many different genres.

  - If you haven’t sampled many different varieties then you can’t have a sharpened judgment.

- Force multipliers can be good or bad.

  - It depends on the underlying thing they’re force multiplying: is it good or bad.

  - Force multiplying just gives *more* to the underlying vector.

  - LLMs are force multipliers.

  - Creative, “yes, and” people are force multipliers.

  - That’s why it’s especially important to have taste and curation in inputs to be force multiplied.

- If you're only going to build one expertise, you have to be the very best at it.

  - Compare that to if you specialize in two or three things that are not typically combined, you can easily be the best at the intersection.

  - Differentiation via *combination*.

- Small companies can be crazy and fast.

  - Crazy because they have no downside, only upside.

  - Fast because they have practically zero coordination costs.

  - Sometimes crazy is how you break through to the next thing.

  - The vast majority of crazy, fast companies are like sparks flying through the wind, snuffing out before anybody notices.

  - But every so often one falls on a patch of dry tinder and *boom*.

- Smaller companies are better at shipping simpler products.

  - E.g. APIs that require fewer clickops to sign up for and use.

  - People often assume that this is because PMs at big companies lose the skill for simplicity.

  - But I think that’s wrong.

  - It’s a thing that’s very easy at a small company when there’s little downside.

  - It’s extremely hard to do at a large established company where there’s lots of downside.

  - It’s not that the people necessarily forget how to do it, it’s that the situation makes it significantly harder.

  - The PMs at smaller companies are simply playing on easy in that dimension.

- It’s not possible to build an open ecosystem on top of a closed one.

  - Imagine creating an open ecosystem on top of the Facebook API that is fundamentally enabled by it.

  - At any point though the powers that be at Facebook could say, “you know, if we were to hit this one button, your whole thing would die. So why don’t you just do whatever we tell you to do?”

  - Related to the ancient Greek notion of the person who can destroy a thing is who truly controls it.

- A downside of centralization of power that we don't think about much: there’s one neck to choke.

  - If you’re in a situation with rule of law and a healthy democracy, then it's not that big of a deal, because the head of state won’t choke anyone’s neck, except on behalf of the machine of government that is the emergent will of the electorate.

  - But if you were in a situation with a head of state that reveled in wanton retribution over perceived slights and had demonstrated their willingness to do it to the centralized companies, that would be a wildly different thing.

  - One CEO of an aggregator rolling over and kissing a ring instantly throws a pall over the whole ecosystem of the aggregator.

  - Aggregator’s ecosystems are not open, but closed.

    - They seem open, but the aggregator ultimately has the power to determine what they allow.

- What you want and what you want to want are often fundamentally at odds.

  - That’s how you can get trapped in a monkey trap, following your short term interest and never overriding it in favor of your long term interest.

- Everyone says they want disconfirming evidence, because they know that's what they're supposed to want to want.

  - But most of the time, people don’t actually want that.

- Capitalism can only give you what you want, not what you *want* to want.

  - What you want to want is often what you *need.*

  - Capitalism allows you to choose what to buy.

  - The incentive is for the creators to figure out what you will most want to buy.

    - Fat, sugar, memes.

    - Supernormal stimuli.

  - What you need is often not what you want.

- The stock market and social media both make things hyper legible in an addictive, all-consuming way.

  - Pulls people towards a short-term optimizing mindset.

  - What you want, not what you want to want.

- Meme-coins are like a hyper caustic acid.

  - They are the ultimate in legibility; a combination of the stock market and social media.

  - They are so potent that they can make it through any communication channel, even low bandwidth, high friction ones.

  - They’ll find a way into the discourse if there’s even the tiniest crack of an opening.

  - I imagine them like *The Shining*: “Heeeeere’s Johnny!”

- If you anthropomorphize an emergent phenomena you will never understand it.

  - You can’t go anywhere from that category error of a singular intelligence applied to what is fundamentally a swarm intelligence.

- The outcome of disruptive changes are only obvious in retrospect.

  - Once it happens it will feel obvious, but also we will see that there were seeds of the future already growing before the disruption.

  - But we don't know which ones are the seeds of the future and which are noise until after!

  - Even a lot of people in the industry who are thinking a lot about LLMs are implicitly thinking about them as simply a sustaining technology (though they *think* they're thinking about things disruptively).

  - Disruptive things are not just *more* than sustaining things, they are *different*.

  - Extrapolation doesn't work for disruptive things, but it does work for sustaining things.

- Sustaining innovations are laminar flow.

  - Disruptive innovations create turbulent flow.

  - Chaos. Impossible to see through the cloud of uncertainty.

  - Once you get to the other side it will feel obvious, inescapable, and you’ll be able to select the random zealots who were ahead of their time (and forget about the 99% that were wrong).

- When everyone thinks the thing can go to the moon they're willing to believe.

  - They'll put up with anything.

  - But when everyone can see the ceiling, they look around and see all the inescapable BS around them and they come crashing back to earth.

    - "If this is all there is, is it worth all of the frustration?"

  - A difference between focusing on the stars or focusing on the quotidian problems of the everyday.

  - One of the reasons momentum fixes all problems.

  - The extrapolation of momentum is to the moon, to infinite value, where every minor annoyance fades away.

- Old stuff people still care about tends to be good stuff.

  - If you’re still talking about it and it’s old it’s Lindy.

  - By default we forget about things as they recede into the distance.

  - If someone keeps it alive, keeps on pulling it to the present because they find it useful, that's a great sign it matters.

  - If it's been pulled into the future from a very long time in the past, that's an incredibly strong signal that it's valuable and interesting.

- How clear and compelling a theory sounds doesn’t matter.

  - All that matters is that it’s viable in the real world.

  - Sometimes beautiful theories don’t work in the real world.

  - Sometimes messy, arbitrary-seeming theories work great.

  - The most sublime is when you somehow get both.

  - But if you can only get one, an ugly but viable theory is way more important.

- After a particularly challenging learning environment, often the learnings feel obvious.

  - You’ve battled through a chaotic environment and emerged much wiser.

  - But when you go to write down or communicate what you learned, everything you can think to say is things you already knew, or were obvious.

    - If you had read it in a book you’d say “well, duh.”

  - Book learning is wildly different from experiential learning.

  - There is no way to substitute book learning for experience.

  - It's one thing to intellectually know something, it's another to feel it in your bones.

  - You can only feel it in your bones from experience.

  - The best way to learn is not by thinking, but by doing.

  - The only exception is if you take the time to think *after* the doing.

    - To squeeze out as much insight from your experiences as you can.

- America is so used to being the default exporter of culture.

  - America has a positive pressure differential of culture that feels like all culture goes out, not in.

  - America won't even know how to reason about what happens when there's an import of culture, they won't even realize what it feels like... or why you might want to be nervous about it.

  - … The DeepSeek models are pretty nifty, though!

- A team is lost when the business team comes up with the idea to charge \$20 a month, and then says, "product team you come up with ideas that are compelling enough that people would pay \$20 a month."

  - The hard part is not deciding to charge \$20.

  - The hard part is something that people want to pay for.

- A phrase I heard this week I found evocative: a “demon seed.”

  - A thing with awesome, overwhelming potential within it, but currently packed up into a little, unassuming, dormant seed.

  - In the right environment it would blossom explosively and change the world.

  - This trope shows up in horror often: the evil spellbook sitting there latent, waiting for someone to open it.

  - The overwhelming power and potential need not be evil or malevolent, it could just be *awesome*, in the original sense of the word.

- Coordination happens no faster than the speed of communication.

  - The faster the communication in the medium, the larger the scale of coordination that is possible, and the quicker coordinated behaviors can appear and adapt.

  - In practice though you don't get more large-scale coordinated things at any given timestep, but more of an ever-adapting cacophony, where various mobs appear and then crash away before you even realize they're there.

  - A power law of size of coordinated things, where the top end gets bigger but the tail still remains.

- The speed of communication changes the tradeoff between speaking and thinking.

  - Transmission vs sense-making.

  - Before the telegraph, everyone communicated by letter.

  - It was slow, which meant that information traveled slowly.

  - But also each letter writer had longer to chew on the information and reflect on it before sending a letter.

  - Now, sending is so easy that we do it immediately and think later… or not at all.

  - Information ricochets around creating a cacophony of undigested information.

  - The sense-making happens emergently at the layer of the collective, which will likely come to different conclusions than if the sense-making happened in human heads and then was shared.

- Everyone’s experienced someone on a video conference struggling with their AirPods and audio.

  - It’s an example of ambient computing; your various computing devices working together seamlessly as one computing experience.

  - When it goes right, it seems magical, and you don’t give it another thought.

  - When it goes wrong, it’s mystifying and completely unclear what to do.

    - Where do you even go to fix it?

  - Wires are clunky, but they are unambiguous about the intent of what should connect to what.

  - If a wired connection isn’t what you want, you have a direct and obvious affordance about what to grab onto and reconfigure.

- In weird times, weirdos reign supreme.

- Interesting means you want to dig deeper.

  - Surprising in a way you find intriguing.

- A framework is rigid and long-lived, whereas a scaffold is temporary support.

- This week I learned about the [<u>Kishõtenketsu story structure</u>](https://en.wikipedia.org/wiki/Kish%C5%8Dtenketsu).

  - It’s an alternate story structure, unlike the hero’s journey

  - No hero, no villain.

  - Just twist.

  - Urban legends often have this form.

- I love this frame on [<u>Organizational Jazz</u>](https://www.linkedin.com/pulse/organizational-jazz-new-ways-work-tom-winans-y6efc/) from John Seely Brown and his collaborators.

- Two very different stances: cynical idealist and pragmatic optimist.

  - Cynical idealist: "This system can never be perfect, because it is fundamentally, irreparably flawed. So what's the point in trying to improve it? We should find another one that could be perfect and not bother with this one."

  - Pragmatic optimist: "This system can never be perfect, because perfection is impossible. But it's pretty good, and with effort we can make it better."

  - The main difference is: do you keep pushing to improve?

  - At any given time step are you default stopping or default pushing?

- A friend asked me if Drucker’s *The Effective Executive* is more Saruman, Radagast, or Gandalf like.

  - I thought Claude’s answer was pretty astute, especially the last line:

  - "I would classify Drucker as predominantly Gandalf-like, with a slight lean toward the Radagast end of the spectrum. While his work is often interpreted through a Saruman lens (especially by those looking to extract simple management principles), his deeper philosophy is about creating environments where people and organizations can flourish naturally. … Radagast/Gandalf wisdom often gets reinterpreted through a Saruman lens because it's easier to package and sell that way."

- I loved [<u>this video explaining the Free Energy Principle</u>](https://www.youtube.com/watch?si=u4wISAoqxtlrHbSA&v=iPj9D9LgK2A&feature=youtu.be) by Artem Kirsanov.

  - The Free Energy Principle is the notion that our minds are constantly making predictions about what we’ll observe, trying to minimize the amount of disagreement between what we predict and what actually happens.

  - The differential is the surprise, the novelty.

  - This is one of the reasons we pay so much attention to novelty; by understanding it we can improve our predictions.

- A culture jamming attack that takes advantage of the Free Energy Principle.

  - Flood the zone with shit, to make the background noise an unpredictable cacophony.

  - Everyone will constantly be burning significant mental cycles trying to make sense of what is happening, which tires everyone out.

  - It then becomes much easier to sneak nefarious things through right under people’s noses because of information overload.

  - I worry that the worst among us have figured out this attack’s potency in the modern information landscape and are shameless enough to use it.

- Sarumans see something that many people miss: that with the right charisma and shamelessness, social reality is surprisingly, near-infinitely malleable.

  - Sarumans who have only fought static forces of nature, like the law of gravity, will conclude that as long as you understand a few basic laws, physical reality will become unimportant and only social reality will remain.

  - But that's wrong.

  - Because physical reality also has a complex adaptive character, and it doesn't care a single iota about your social reality.

  - Phenomena like the economy where each individual makes highly local and self-optimizing decisions that add up to massive tidal waves, will knock the Saruman back down, hard.

- "Con artist" means confidence artist.

  - They earn your trust and confidence and then abuse it.

  - One common way to do this is they literally speak with confidence and without shame which makes you trust them.

    - “If they weren’t sure they wouldn’t sound so confident.”

  - They could have no shame because they deeply and authentically believe the thing they're confidently asserting... or because they are incapable of feeling shame.

- Mass protests erupt when there’s a general anger simmering and then an inciting incident causes a cascade.

  - The inciting incident forms a schelling point that has to stand out as obviously beyond the pale to get to the critical mass where the compounding anger can accumulate on its own.

  - But a flood the end zone with shit strategy aims for a confusing background mess.

  - Nothing stands out as obviously the thing to coordinate around because there’s lots of candidates.

  - It’s like a [<u>three stooge syndrome</u>](https://simpsons.fandom.com/wiki/Three_Stooges_Syndrome) model of gumming up coordination.

  - Coordination happens not due to the absolute magnitude of the signal but the [*<u>prominence</u>*](https://en.wikipedia.org/wiki/Topographic_prominence) of it, the differential strength of it compared to alternatives.

  - The one bit of good news is that if the anger is strong enough a cascade is inevitable, just hard to predict when it will explode onto the scene.

- Most people I talk with think I agree with them.

  - That’s partially because to understand others’ perspectives I work to steelman their ideas back to them to make sure I get it.

  - But it’s also because I have a relentless “yes, and” kind of energy.

  - You will rarely hear the word “no” from me.

  - In conversations I'm finding the part I do agree with (perhaps a very small part) and emphatically agreeing on that part.

  - But they can miss the *absence* of a “yes” easier than the *presence* of a “no”.

  - They miss what I'm *not* emphatically agreeing on.

  - However, they are right in one thing: I think that everyone is inherently reasonable, and understanding how the world looks from their eyes makes everyone better off.